# Global Transport Couplings for Classifier-Free Guided Flows

Katarina Petrović<sup>1,2</sup> Zander W. Blasingame<sup>1</sup> Danyal Rehman<sup>1,3,4,5</sup> İsmail İlkan Ceylan<sup>6,1,2</sup> Michael Bronstein<sup>1,2</sup> Stephen Y. Zhang<sup>7†</sup> Lazar Atanackovic<sup>8,9,10†</sup> Alexander Tong<sup>1†</sup>

<sup>1</sup>AITHYRA <sup>2</sup>University of Oxford <sup>3</sup>Mila - Québec AI Institute <sup>4</sup>Université de Montréal <sup>5</sup>Massachusetts Institute of Technology <sup>6</sup>Technische Universität Wien <sup>7</sup>Flatiron Institute <sup>8</sup>University of Alberta <sup>9</sup>Alberta Machine Intelligence Institute <sup>10</sup>Canada CIFAR AI Chair

<sup>†</sup>Equal advising

## Abstract

Optimal-transport couplings have been shown to reduce training variance in unconditional flow models, but their role in conditional generation remains unclear. A natural approach constructs separate couplings for each condition, but this is impractical for large or continuous conditioning spaces found in modern image foundation models. We introduce Global Transport (GT), a global class-agnostic optimal-transport coupling, computed without class labels. GT can associate different conditions with different regions of the source noise, and consequently worsens performance without guidance. However, when combined with classifier-free guidance (CFG), GT consistently improves generation across domains, model scales, and sampling budgets. This reversal suggests that couplings for conditional flows should be evaluated both empirically and theoretically under the guided flow used at inference, rather than on unguided generation. We evaluate GT over both discrete class and continuous text conditioned image generation across model scales, and investigate how coupling choice alters guided trajectories. These results identify coupling design in the guided flow setting as a simple training time axis to improve performance without modifying existing architectures, samplers, or guidance mechanisms.

Correspondence: kpetrovic@aithyra.at Date: 6 October, 2026

## 1 Introduction

Flow matchin<sub>g</sub> (Alber<sub>g</sub>o and Vanden-Eijnden, 2023; Li<sub>p</sub>man et al., 2023; Peluchetti, 2023) and difusion models (Ho et al., 2020; Son<sub>g</sub> et al., 2021) have emer<sub>g</sub>ed as <sub>p</sub>owerful <sub>p</sub>aradi<sub>g</sub>ms for <sub>g</sub>enerative modelin<sub>g</sub>, achievin state-of-the-art eneration results across a wide ran e of domains includin ima es (Rombach et al., 2022; Esser et al., 2024; Ma et al., 2024), videos (Ho et al., 2022; Blattmann et al., 2023; Pol<sub>y</sub>ak et al., 2024), and biolo<sub>g</sub>ical data (Li et al., 2026; Morehead et al., 2026). In the sim<sub>p</sub>lest construction, source and data sam<sub>p</sub>les are <sub>p</sub>aired inde<sub>p</sub>endentl<sub>y</sub>. A widel<sub>y</sub> ado<sub>p</sub>ted strate<sub>gy</sub> for im<sub>p</sub>rovin<sub>g g</sub>eneration <sub>q</sub>ualit<sub>y</sub> is b<sub>y</sub> incor<sub>p</sub>oratin<sub>g</sub> o<sub>p</sub>tima transport (OT) couplings between noise samples and data points (Pooladian et al., 2023; Tong, Fatras, et al., 2024). OT <sub>p</sub>airs noise and data to minimize trans<sub>p</sub>ort cost, reducin<sub>g</sub> the trainin<sub>g</sub> variance and enablin<sub>g</sub> more eficient numerical inte<sub>g</sub>ration and sam<sub>p</sub>lin<sub>g</sub> eficienc<sub>y</sub> of flow matchin<sub>g</sub> models (Berthelot et al., 2026; Lu, Sun, et al., 2026; Malnick et al., 2026). These benefits su<sub>gg</sub>est that cou<sub>p</sub>lin<sub>g</sub> desi<sub>g</sub>n could also im<sub>p</sub>rove conditional <sub>g</sub>eneration. It is less obvious<sub>,</sub> however<sub>,</sub> what an a<sub>pp</sub>ro<sub>p</sub>riate cou<sub>p</sub>lin<sub>g</sub> should be when ever<sub>y</sub> data sam<sub>p</sub>le is associated with a class<sub>,</sub> text <sub>p</sub>rom<sub>p</sub>t<sub>,</sub> or other condition.

Th<sub>e</sub> n<sub>a</sub>t<sub>u</sub>r<sub>a</sub>l <sub>e</sub>xt<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n <sub>o</sub>f thi<sub>s pa</sub>r<sub>a</sub>di<sub>g</sub>m i<sub>s</sub> t<sub>o</sub> m<sub>a</sub>t<sub>c</sub>h <sub>sou</sub>r<sub>ce a</sub>nd d<sub>a</sub>t<sub>a sepa</sub>r<sub>a</sub>t<sub>e</sub>l<sub>y</sub> f<sub>o</sub>r <sub>eac</sub>h <sub>co</sub>nditi<sub>o</sub>n<sub>.</sub> S<sub>uc</sub>h <sub>c</sub>l<sub>ass</sub>- <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l OT r<sub>espec</sub>t<sub>s</sub> th<sub>e co</sub>mm<sub>o</sub>n <sub>sou</sub>r<sub>ce</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>use</sub>d t<sub>o sa</sub>m<sub>p</sub>l<sub>e eac</sub>h <sub>c</sub>l<sub>ass,</sub> b<sub>u</sub>t r<sub>equ</sub>ir<sub>es su</sub>fi<sub>c</sub>i<sub>e</sub>ntl<sub>y</sub> l<sub>a</sub>r<sub>ge</sub> batches of exam<sub>p</sub>les sharin<sub>g</sub> a condition (Kerri<sub>g</sub>an et al., 2024; Chemseddine et al., 2025; Chen<sub>g</sub> and Schwin<sub>g</sub>, 2025; Kon<sub>g</sub> et al., 2026; Mousavi-Hosseini et al., 2026). This becomes im<sub>p</sub>ractical when there are man<sub>y</sub> classes <sub>a</sub>nd i<sub>s</sub> inf<sub>eas</sub>ibl<sub>e</sub> wh<sub>e</sub>n <sub>co</sub>nditi<sub>o</sub>n<sub>s,</sub> <sub>suc</sub>h <sub>as</sub> t<sub>e</sub>xt-<sub>e</sub>mb<sub>e</sub>ddin<sub>gs,</sub> <sub>a</sub>r<sub>e</sub> <sub>co</sub>ntin<sub>uous.</sub> A<sub>s</sub> <sub>a</sub> r<sub>esu</sub>lt<sub>,</sub> <sub>c</sub>l<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l OT has achieved limited ado<sub>p</sub>tion in the conditional <sub>g</sub>enerative settin<sub>g</sub>.

Th<sub>e</sub>r<sub>e</sub> i<sub>s,</sub> h<sub>o</sub>w<sub>e</sub>v<sub>e</sub>r<sub>, a</sub>n <sub>a</sub>dditi<sub>o</sub>n<sub>a</sub>l <sub>co</sub>nf<sub>ou</sub>nd<sub>e</sub>r in th<sub>e case o</sub>f <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> C<sub>o</sub>nditi<sub>o</sub>n<sub>a</sub>l fl<sub>o</sub>w m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> are almost ubi<sub>q</sub>uitousl<sub>y</sub> sam<sub>p</sub>led with classifier-free <sub>g</sub>uidance (CFG) (Ho and Salimans, 2022), which combines <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>a</sub>nd <sub>u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l v<sub>e</sub>l<sub>oc</sub>iti<sub>es a</sub>nd i<sub>s use</sub>d t<sub>o s</sub>t<sub>ee</sub>r t<sub>o</sub>w<sub>a</sub>rd<sub>s a co</sub>nditi<sub>o</sub>n <sub>a</sub>nd <sub>a</sub>w<sub>ay</sub> fr<sub>o</sub>m th<sub>e u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l d<sub>e</sub>n<sub>s</sub>it<sub>y.</sub> Cl<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l OT h<sub>as</sub> b<sub>ee</sub>n <sub>s</sub>h<sub>o</sub>wn t<sub>o</sub> im<sub>p</sub>r<sub>o</sub>v<sub>e</sub> b<sub>o</sub>th th<sub>e u</sub>n<sub>gu</sub>id<sub>e</sub>d <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fl<sub>o</sub>w<sub>, a</sub>nd th<sub>e</sub> guided flow (Cheng and Schwing, 2025). The natural extension of this paradigm to the conditional generative setting is via class-conditional OT, where one seeks an optimal coupling for each class. However, this procedure n<sub>ecess</sub>it<sub>a</sub>t<sub>es</sub> th<sub>e co</sub>m<sub>pu</sub>t<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f <sub>coup</sub>lin<sub>gs</sub> f<sub>o</sub>r <sub>c</sub>l<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>e</sub>d b<sub>a</sub>t<sub>c</sub>h<sub>es,</sub> whi<sub>c</sub>h b<sub>eco</sub>m<sub>es</sub> im<sub>p</sub>r<sub>ac</sub>ti<sub>ca</sub>l f<sub>o</sub>r l<sub>a</sub>r<sub>ge</sub> conditionin<sub>g</sub> s<sub>p</sub>aces fre<sub>q</sub>uentl<sub>y</sub> found in modern <sub>g</sub>enerative models<sub>,</sub> and infeasible in continuous conditionin<sub>g</sub> s<sub>p</sub>aces used in text-to-ima<sub>g</sub>e models (Esser et al., 2024). As a result, class-conditional OT has achieved limited ado<sub>p</sub>tion in the conditional <sub>g</sub>enerative settin<sub>g,</sub> and <sub>y</sub>ields modest <sub>g</sub>ains for the added com<sub>p</sub>utational cost.

CFG trains a flow model to <sub>p</sub>erform both conditional and unconditional <sub>g</sub>eneration<sub>,</sub> dro<sub>pp</sub>in<sub>g</sub> the conditionin<sub>g</sub> si<sub>g</sub>nal on some fraction of the data <sub>p</sub>oints<sub>;</sub> at inference time<sub>,</sub> <sub>g</sub>uided sam<sub>p</sub>les are drawn b<sub>y</sub> linearl<sub>y</sub> com<sub>p</sub>osin<sub>g</sub> the two velocit fields modulated b a uidance scale. As a result CFG uided sam les follow trajectories defined jointly by both conditional and unconditional velocity fields, rather than by the conditional model alone. Cl<sub>ass co</sub>nditi<sub>o</sub>n<sub>a</sub>l OT im<sub>p</sub>r<sub>o</sub>v<sub>es</sub> b<sub>o</sub>th th<sub>e u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>a</sub>nd <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fi<sub>e</sub>ld<sub>s sepa</sub>r<sub>a</sub>t<sub>e</sub>l<sub>y.</sub> A n<sub>a</sub>t<sub>u</sub>r<sub>a</sub>l <sub>ques</sub>ti<sub>o</sub>n th<sub>en ar</sub>i<sub>ses:</sub>

## Can a coupling that considers both fields together produce a better guided generator?

We answer in the afirmative with Global Transport (GT), which computes a single global OT assignment across source and target samples without considering their conditions. GT is straightforward to apply to both class labels and text conditioning, and requires no change to the model, sampler, or CFG rule. However, GT comes with a tr<sub>a</sub>d<sub>eo</sub>f<sub>:</sub> Alth<sub>oug</sub>h th<sub>e coup</sub>lin<sub>g p</sub>r<sub>ese</sub>rv<sub>es</sub> th<sub>e o</sub>v<sub>e</sub>r<sub>a</sub>ll <sub>sou</sub>r<sub>ce</sub> m<sub>a</sub>r<sub>g</sub>in<sub>a</sub>l<sub>,</sub> th<sub>e co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>sou</sub>r<sub>ce</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n d<sub>o</sub> n<sub>o</sub>t need to be consistent with the unconditional source distribution. Following from this mismatch, GT performs worse in unguided conditional sampling in our experiments. Under CFG, however, the ordering reverses with GT im<sub>p</sub>rovin<sub>g g</sub>uided <sub>g</sub>eneration across domains<sub>,</sub> model scales<sub>,</sub> and sam<sub>p</sub>lin<sub>g</sub> bud<sub>g</sub>ets.

To investigate the cause of this reversal, we study the coupling choice from the perspective of the guided flow. We find GT reduces the prediction gap (Wang et al., 2025; Cai et al., 2026), the diference between the conditional and unconditional flows. This reduces a pattern we term the yo-yo efect, where independent flows exhibit pronounced contraction, re-expansion, and overshooting along guided trajectories with increasing guidance stren<sub>g</sub>th. Our contributions are:

• We propose GT, a practical, low-overhead coupling strategy for conditional generation that requires no change to architectures<sub>,</sub> sam<sub>p</sub>lers<sub>,</sub> or <sub>g</sub>uidance mechanisms.

• W<sub>e</sub> d<sub>e</sub>m<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>t<sub>e</sub> th<sub>a</sub>t <sub>u</sub>n<sub>gu</sub>id<sub>e</sub>d <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> d<sub>oes</sub> n<sub>o</sub>t <sub>a</sub>lw<sub>ays co</sub>rr<sub>e</sub>l<sub>a</sub>t<sub>e</sub> with <sub>gu</sub>id<sub>e</sub>d <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce,</sub> and analyze how coupling choice afects guided trajectories and the prediction gap.

• We show that GT consistently yields competitive generation performance on class-conditioned image <sub>g</sub>eneration (e.<sub>g</sub>. FID 1.91 on Ima<sub>g</sub>eNet-256), text-conditioned ima<sub>g</sub>e <sub>g</sub>eneration, and sin<sub>g</sub>le-cell <sub>g</sub>ene <sub>e</sub>x<sub>p</sub>r<sub>ess</sub>i<sub>o</sub>n d<sub>a</sub>t<sub>a, ac</sub>r<sub>oss</sub> m<sub>o</sub>d<sub>e</sub>l <sub>sca</sub>l<sub>es a</sub>nd <sub>sa</sub>m<sub>p</sub>lin<sub>g</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>s, a</sub>nd <sub>ou</sub>t<sub>pe</sub>rf<sub>o</sub>rm<sub>s</sub> b<sub>o</sub>th ind<sub>epe</sub>nd<sub>e</sub>nt <sub>a</sub>nd class-conditional cou<sub>p</sub>lin<sub>g</sub>s.

## 2 Background and preliminaries

Flow Matching. Flow matching (FM) (Albergo et al., 2023; Lipman et al., 2023; Liu et al., 2023; Peluchetti, 2023) allows continuous time trans<sub>p</sub>ort between distributions: the al<sub>g</sub>orithm trains a velocit<sub>y</sub> field that evolves <sub>sa</sub>m<sub>p</sub>l<sub>es</sub> fr<sub>o</sub>m <sub>a sou</sub>r<sub>ce</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $p _ { 0 }$ to a tar<sub>g</sub>et data distribution $p _ { 1 } = p _ { \mathrm { d a t a } }$ <sub>, w</sub>h<sub>ere</sub> $p _ { 0 }$ is t<sub>yp</sub>icall<sub>y</sub> chosen to b<sub>e a</sub>n <sub>easy</sub>-t<sub>o</sub>-<sub>sa</sub>m<sub>p</sub>l<sub>e</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>suc</sub>h <sub>as</sub> $\mathcal { N } ( 0 , I _ { d } )$

Posit a continuous-time transport, i.e. a probability path, $( p _ { t } ) _ { t \in [ 0 , 1 ] }$ with th<sub>e</sub> r<sub>esc</sub>rib<sub>e</sub>d b<sub>ou</sub>nd<sub>a</sub>r <sub>co</sub>ndi ti<sub>o</sub>n<sub>s</sub> $( p _ { 0 } , p _ { 1 } )$ <sub>.</sub> An <sub>suc</sub>h $p _ { t }$ i<sub>s ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d b <sub>a</sub> m<sub>a</sub>r<sub>g</sub>in<sub>a</sub>l v<sub>e</sub>l<sub>oc</sub>it fi<sub>e</sub>ld $u _ { t }$ <sub>suc</sub>h th<sub>a</sub>t $\partial _ { t } p _ { t } + \nabla \cdot \left( p _ { t } \ * u _ { t } \right) \ = \ 0 .$ FM seeks to a<sub>pp</sub>roximate $u _ { t }$ <sub>w</sub>ith <sub>a mo</sub>d<sub>e</sub>l $u _ { t } ^ { \theta }$ b<sub>y</sub> least-s<sub>q</sub>uares re<sub>g</sub>ression<sub>,</sub> minimizin<sub>g</sub> $\begin{array} { r l } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) } & { { } = } \end{array}$ $\mathbb { E } _ { t \sim U [ 0 , 1 ] , x _ { t } \sim p _ { t } } \left\lceil \left\| u _ { t } ^ { \theta } ( x _ { t } ) - u _ { t } ( x _ { t } ) \right\| _ { 2 } ^ { 2 } \right\rceil$ This objective is intractable, as in practice $u _ { t }$ i<sub>s u</sub>n<sub>a</sub>v<sub>a</sub>il<sub>a</sub>bl<sub>e</sub> in closed form. Conditional FM (Alber<sub>g</sub>o et al., 2023; Ton<sub>g</sub>, Fatras, et al., 2024) instead introduces a cou<sub>p</sub>lin<sub>g</sub> $\pi \in \Pi ( p _ { 0 } , p _ { 1 } )$ <sub>a</sub>nd <sub>a co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>p</sub>r<sub>o</sub>b<sub>a</sub>bilit<sub>y pa</sub>th $p _ { t } ( \cdot \mid x _ { 0 } , x _ { 1 } )$ <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d b<sub>y</sub> tr<sub>ac</sub>t<sub>a</sub>bl<sub>e co</sub>nditi<sub>o</sub>n<sub>a</sub>l v<sub>e</sub>l<sub>oc</sub>it<sub>y</sub> fi<sub>e</sub>ld $\boldsymbol { u } _ { t } ( \cdot | \boldsymbol { x } _ { 0 } , \boldsymbol { x } _ { 1 } )$ . Taking expectations yields the CFM objective, which trains a neural velocity field $u ^ { \theta }$ <sup>b</sup>y regression <sub>on</sub>t<sub>o</sub> th<sub>e c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm con</sub>diti<sub>ona</sub>l <sub>ve</sub>l<sub>oc</sub>it $u _ { t } ( x _ { t } \mid x _ { 0 } , x _ { 1 } )$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , ( x _ { 0 } , x _ { 1 } ) \sim \pi , x _ { t } \sim p _ { t } ( \cdot | x _ { 0 } , x _ { 1 } ) } \left[ \left| \left| u _ { t } ^ { \theta } ( x _ { t } ) - u _ { t } ( x _ { t } \mid x _ { 0 } , x _ { 1 } ) \right| \right| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{1}
$$

On<sub>ce</sub> tr<sub>a</sub>in<sub>e</sub>d<sub>, sa</sub>m<sub>p</sub>l<sub>es</sub> fr<sub>o</sub>m $p _ { 1 }$ <sub>ca</sub>n b<sub>e ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d b<sub>y</sub> fir<sub>s</sub>t dr<sub>a</sub>win<sub>g</sub> fr<sub>o</sub>m th<sub>e sou</sub>r<sub>ce</sub> $x _ { 0 } \sim p _ { 0 }$ <sub>,</sub> th<sub>en</sub> <sub>numer</sub>i<sub>ca</sub>ll<sub>y</sub> inte<sub>g</sub>ratin<sub>g</sub> the ODE ${ \dot { x } } _ { t } = u _ { t } ( x _ { t } )$

Classifier-free Guidance. Conditional generation, present in many applications such as text-to-image or textto-video s<sub>y</sub>nthesis<sub>,</sub> re<sub>q</sub>uires sam<sub>p</sub>lin<sub>g</sub> <sub>g</sub>iven a <sub>p</sub>rom<sub>p</sub>t $c \in { \mathcal { C } }$ <sub>,</sub> achieved b<sub>y</sub> learnin<sub>g</sub> a <sub>p</sub>rom<sub>p</sub>t-conditioned velocit<sub>y</sub> fi<sub>e</sub>ld $\boldsymbol { u } _ { t } ( \boldsymbol { x } _ { t } \mid c )$ via the analo<sub>g</sub>ue of e<sub>q</sub>uation 1<sub>,</sub>

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ^ { c } ( \theta ) = \mathbb { E } _ { t , ( x _ { 0 } , x _ { 1 } , c ) \sim \pi , x _ { t } \sim p _ { t } ( \cdot | x _ { 0 } , x _ { 1 } ) } \left[ \left| \left| u _ { t } ^ { \theta } ( x _ { t } , c ) - u _ { t } ( x _ { t } \mid x _ { 0 } , x _ { 1 } , c ) \right| \right| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{2}
$$

Classifier-free <sub>g</sub>uidance (CFG) (Ho and Salimans, 2022) im<sub>p</sub>roves sam<sub>p</sub>le <sub>q</sub>ualit<sub>y</sub> and <sub>p</sub>rom<sub>p</sub>t ali<sub>g</sub>nment b<sub>y</sub> <sub>sa</sub>m<sub>p</sub>lin<sub>g</sub> fr<sub>o</sub>m <sub>a</sub> lin<sub>ea</sub>r <sub>co</sub>mbin<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> m<sub>a</sub>r<sub>g</sub>in<sub>a</sub>l <sub>a</sub>nd <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fi<sub>e</sub>ld<sub>s,</sub>

$$
\begin{array} { r } { \widetilde { u } _ { t } ^ { w } ( x _ { t } \mid c ) = u _ { t } ( x _ { t } ) + w g ( x _ { t } , t , c ) , \quad g ( x _ { t } , t , c ) = u _ { t } ( x _ { t } \mid c ) - u _ { t } ( x _ { t } ) } \end{array}\tag{3}
$$

<sub>w</sub>h<sub>ere</sub> $w \ge 1$ <sub>a</sub>nd $g$ are the <sub>g</sub>uidance wei<sub>g</sub>ht and vector res<sub>p</sub>ectivel<sub>y</sub>. In <sub>p</sub>ractice<sub>,</sub> a sin<sub>g</sub>le model is trained across all conditions, with c replaced by a null token ∅ with some probability, so that $u _ { t } ^ { \theta } ( x _ { t } , \theta )$ estimates the mar<sub>g</sub>ina field. A<sub>pp</sub>endix E.1 reviews the theor<sub>y</sub> underl<sub>y</sub>in<sub>g</sub> CFG.

Optimal Transport. Optimal transport conditional flow matching (OT-CFM) (Pooladian et al., 2023; Tong, Fatras, et al., 2024; Kon<sub>g</sub> et al., 2026; Mousavi-Hosseini et al., 2026) re<sub>p</sub>laces the inde<sub>p</sub>endent cou<sub>p</sub>lin<sub>g</sub> $\pi = p _ { 0 } \otimes p _ { 1 }$ with a roximations of the Euclidean OT cou lin π , defined as the solution to

$$
\begin{array} { r } { \pi _ { \mathrm { O T } } = \mathop { \bf a r g m i n } _ { \pi \in \Pi ( p _ { 0 } , p _ { 1 } ) } \mathbb { E } _ { ( x _ { 0 } , x _ { 1 } ) \sim \pi } \| x _ { 0 } - x _ { 1 } \| _ { . } ^ { 2 } } \end{array}\tag{4}
$$

The resulting trajectories are straighter in practice and therefore easier to integrate in the few-step sampling re<sub>g</sub>ime<sub>,</sub> shown to achieve modest but consistent <sub>g</sub>ains in the unconditional settin<sub>g</sub>.

Class-conditional Optimal Transport. Class-conditional eneration involves sam lin from data distribution $p _ { 1 } ( x \mid c )$ <sub>co</sub>nditi<sub>o</sub>n<sub>e</sub>d <sub>o</sub>n <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>c</sub>l<sub>ass</sub> $c \in { \mathcal { C } } .$ A n<sub>a</sub>ïv<sub>e a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f minib<sub>a</sub>t<sub>c</sub>h OT-CFM t<sub>o</sub> th<sub>e c</sub>l<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l setting involves solving |C| independent OT problems, one for each class $c \in { \mathcal { C } }$ . However<sub>,</sub> this <sub>q</sub>uickl<sub>y</sub> becomes infeasible when the number of classes |C| becomes even moderately large (for instance, ImageNet contains 1000 classes), and is not <sub>p</sub>ossible in settin<sub>g</sub>s where the condition itself is a continuous latent state, as is the case for text-to-ima<sub>g</sub>e <sub>g</sub>eneration for exam<sub>p</sub>le. (Kerri<sub>g</sub>an et al., 2024; Chemseddine et al., 2025; Chen<sub>g</sub> and Schwin<sub>g</sub>, 2025) <sub>p</sub>ro<sub>p</sub>osed to a<sub>pp</sub>roximate this b<sub>y</sub> minimizin<sub>g</sub>

$$
\pi _ { \mathbf { C ^ { 2 } O T } } = \arg \operatorname* { m i n } _ { \pi \in \Pi ( p _ { 0 } , p _ { 1 } ) } \mathbb { E } _ { ( ( x _ { 0 } , c _ { 0 } ) , ( x _ { 1 } , c _ { 1 } ) ) \sim \pi } \left[ \| x _ { 0 } - x _ { 1 } \| ^ { 2 } + \beta \| c _ { 0 } - c _ { 1 } \| ^ { 2 } \right]\tag{5}
$$

<sub>w</sub>h<sub>ere</sub> $\beta$ is adjusted to fit the scale and $c _ { 0 , 1 }$ d<sub>e</sub>n<sub>o</sub>t<sub>es</sub> th<sub>e c</sub>l<sub>ass assoc</sub>i<sub>a</sub>t<sub>e</sub>d with $x _ { 0 , 1 }$ <sub>.</sub> Thi<sub>s a</sub>ll<sub>ows</sub> th<sub>e use o</sub>f <sub>muc</sub>h <sub>s</sub>m<sub>a</sub>ll<sub>e</sub>r b<sub>a</sub>t<sub>c</sub>h<sub>es</sub> in <sub>p</sub>r<sub>ac</sub>ti<sub>ce,</sub> b<sub>u</sub>t with <sub>u</sub>nkn<sub>o</sub>wn d<sub>eg</sub>r<sub>a</sub>d<sub>a</sub>ti<sub>o</sub>n<sub>, a</sub>nd th<sub>us</sub> f<sub>a</sub>r h<sub>as</sub> n<sub>o</sub>t b<sub>ee</sub>n wid<sub>e</sub>l<sub>y a</sub>d<sub>op</sub>t<sub>e</sub>d in conditional <sub>g</sub>eneration settin<sub>g</sub>s<sub>,</sub> where inde<sub>p</sub>endent cou<sub>p</sub>lin<sub>g</sub>s are almost universal.

## 3 Global Transport

We next introduce Global Transport (GT) a practical <sub>a</sub>nd l<sub>o</sub>w <sub>o</sub>v<sub>e</sub>rh<sub>ea</sub>d <sub>app</sub>li<sub>ca</sub>ti<sub>o</sub>n <sub>o</sub>f <sub>op</sub>tim<sub>a</sub>l tr<sub>a</sub>n<sub>spo</sub>rt whi<sub>c</sub>h im<sub>p</sub>r<sub>o</sub>v<sub>es pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> in th<sub>e c</sub>l<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>e</sub>d setting. GT computes an approximate optimal trans-<sub>p</sub>ort cou<sub>p</sub>lin<sub>g</sub> without encoura<sub>g</sub>in<sub>g</sub> or constrainin<sub>g</sub> as-<sub>s</sub>i<sub>g</sub>nm<sub>e</sub>nt<sub>s</sub> t<sub>o p</sub>r<sub>ese</sub>rv<sub>e c</sub>l<sub>ass</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s.</sub> Th<sub>e</sub> m<sub>e</sub>th<sub>o</sub>d d<sub>oes</sub>

Algorithm 1 GT training step   
$x _ { 0 } \sim p _ { 0 } , \ ( x _ { 1 } , c ) \sim p _ { 1 } , \ t \sim \mathcal { U } ( 0 , 1 )$   
$\tilde { c } \gets \emptyset$ w.p. p<sub>uncond</sub>, else ${ \tilde { c } } \gets c$   
$\hat { x } _ { 0 } \gets \pi _ { \mathsf { G T } } ( x _ { 0 } ) ; \quad x _ { t } \gets t x _ { 1 } + ( 1 - t ) \hat { x } _ { 0 }$   
$\theta \gets \mathrm { U p d a t e } \big ( \theta , \nabla _ { \theta } \| u _ { t } ^ { \theta } ( x _ { t } , \tilde { c } ) - ( x _ { 1 } - \hat { x } _ { 0 } ) \| ^ { 2 } \big )$

not chan<sub>g</sub>e the architecture nor inference al<sub>g</sub>orithm and adds onl<sub>y</sub> a trainin<sub>g</sub>-time assi<sub>g</sub>nment ste<sub>p</sub>. We first describe the overall objective, then two a<sub>pp</sub>roximations either usin<sub>g</sub> (1) a mini-batch a<sub>pp</sub>roximation (Fatras et al., 2021; Pooladian et al., 2023; Ton<sub>g</sub>, Fatras, et al., 2024) or (2) a semi-discrete a<sub>pp</sub>roximation (Kon<sub>g</sub> et al., 2026; Mousavi-Hosseini et al., 2026).

General objective. Given noise samples $x _ { 0 } \sim p _ { 0 }$ <sub>a</sub>nd <sub>c</sub>l<sub>ass co</sub>nditi<sub>o</sub>n<sub>a</sub>l d<sub>a</sub>t<sub>a sa</sub>m<sub>p</sub>l<sub>es</sub> $( x _ { 1 } , c ) \sim p _ { 1 }$ <sub>,</sub> f<sub>or</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sup>tr</sup>a<sup>n</sup>spo<sup>rt</sup> coup<sup>lin</sup>g $\pi _ { O T }$ <sub>as</sub> d<sub>e</sub>fin<sub>e</sub>d in <sub>equa</sub>ti<sub>o</sub>n <sub>4,</sub> <sub>a</sub>nd <sub>a</sub> <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rd lin<sub>ea</sub>r fl<sub>o</sub>w m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> <sub>pa</sub>th $x _ { t } = t x _ { 1 } + ( 1 - t ) x _ { 0 }$ where the noise is rearranged to match the closest data points, ignoring the labels c. This results in the conditional flow matching objective:

$$
\mathcal { L } _ { \mathrm { G T } } ( \theta ) = \mathbb { E } _ { t \sim U [ 0 , 1 ] , ( x _ { 0 } , x _ { 1 } ) \sim \pi _ { \mathrm { O T } } } \Vert u _ { t } ^ { \theta } ( x _ { t } , \tilde { c } ) - ( x _ { 1 } - x _ { 0 } ) \Vert ^ { 2 }\tag{6}
$$

where c˜ is set to ∅ with probability $p _ { \mathrm { u n c o n d } }$ and c otherwise. The full algorithm appears in Algorithm 1.

Minibatch-OT Implementation. We com ute the assi nment usin the Hun arian al orithm with s uared E<sub>uc</sub>lid<sub>ea</sub>n <sub>cos</sub>t in th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l in <sub>u</sub>t <sub>s ace.</sub> F<sub>o</sub>r <sub>a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>o</sub>f <sub>s</sub>iz<sub>e</sub> $B ,$ th<sub>e cos</sub>t m<sub>a</sub>trix r<sub>e u</sub>ir<sub>es</sub> $O ( B ^ { 2 } )$ <sub>memor an</sub>d th<sub>e</sub> exact ass<sup>i</sup><sub>g</sub>nment $O ( B ^ { 3 } )$ time. This is <sub>g</sub>enerall<sub>y</sub> dwarfed b<sub>y</sub> network evaluation time on lar<sub>g</sub>e s<sub>y</sub>stems and is ne<sub>g</sub>li<sub>g</sub>ible (usuall<sub>y</sub> < 2% overhead) for modern workflows. However, it can be ex<sub>p</sub>ensive for lar<sub>g</sub>e batch sizes, if this settin<sub>g</sub> is desired a re<sub>g</sub>ularized (Cuturi, 2013; Ton<sub>g</sub>, Malkin, et al., 2024; Zhan<sub>g</sub> et al., 2026), or semi-discrete app<sup>r</sup>oac<sup>h m</sup>ay <sup>b</sup>e <sup>m</sup>o<sup>r</sup>e app<sup>r</sup>op<sup>ri</sup>a<sup>t</sup>e.

Semi-discrete OT Implementation. Building of of recent work on Flow matching with semi-discrete optimal trans<sub>p</sub>ort cou<sub>p</sub>lin<sub>g</sub>s<sub>,</sub> we can also use an a<sub>pp</sub>roximate semi-discrete OT im<sub>p</sub>lementation. This a<sub>pp</sub>roach first <sub>p</sub>erforms an ex<sub>p</sub>ensive <sub>p</sub>re<sub>p</sub>rocessin<sub>g</sub> ste<sub>p</sub> to o<sub>p</sub>timize a semi-dual <sub>p</sub>otential function over all discrete data<sub>p</sub>oints<sub>,</sub> whi<sub>c</sub>h <sub>ca</sub>n th<sub>e</sub>n b<sub>e use</sub>d t<sub>o ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e coup</sub>lin<sub>gs</sub> with <sub>a</sub>n<sub>y</sub> m<sub>e</sub>mb<sub>e</sub>r <sub>o</sub>f th<sub>e co</sub>ntin<sub>uous</sub> n<sub>o</sub>i<sub>se</sub> m<sub>easu</sub>r<sub>e.</sub>

The conditional source distribution mismatch. Ignoring conditions while constructing the coupling changes th<sub>e</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>o</sub>f $( x _ { 0 } , x _ { 1 } )$ <sub>pa</sub>i<sub>rs</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>mo</sub>d<sub>e</sub>l l<sub>earns</sub> f<sub>rom.</sub> T<sub>o</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub> thi<sub>s, we cons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>u</sub>ll b<sub>a</sub>t<sub>c</sub>h<sub>,</sub> idealized settin<sub>g</sub> usin<sub>g</sub> the true OT cou<sub>p</sub>lin<sub>g</sub> $\pi _ { O T }$ <sub>,</sub> with associated trans<sub>p</sub>ort ma<sub>p</sub> $T : p _ { 0 } \to p _ { 1 }$ . Draw $x _ { 0 } \sim p _ { 0 }$ <sub>a</sub>nd <sub>se</sub>t $x _ { 1 } = T ( x _ { 0 } )$ with <sub>co</sub>nditi<sub>o</sub>n $C .$ . Then the source points that are paired with points with condition c do n<sub>o</sub>t n<sub>ee</sub>d t<sub>o</sub> b<sub>e</sub> di<sub>s</sub>trib<sub>u</sub>t<sub>e</sub>d lik<sub>e</sub> $p _ { 0 }$ as diferent conditions ma<sub>y</sub> be <sub>p</sub>aired with source <sub>p</sub>oints from diferent re<sub>g</sub>ions <sub>o</sub>f th<sub>e pr</sub>i<sub>or.</sub> I<sub>n o</sub>th<sub>er wor</sub>d<sub>s, a</sub>lth<sub>oug</sub>h $x _ { 0 } \sim p _ { 0 }$ <sub>,</sub> th<sub>e co</sub>nditi<sub>o</sub>n<sub>a</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $x _ { 0 } \mid C = c$ n<sub>ee</sub>d n<sub>o</sub>t <sub>equa</sub>l $p _ { 0 }$ <sub>.</sub> Thi<sub>s</sub> t<sub>e</sub>ll<sub>s</sub> <sub>us</sub> th<sub>a</sub>t <sub>u</sub>nd<sub>e</sub>r th<sub>e</sub> <sub>u</sub>n<sub>gu</sub>id<sub>e</sub>d <sub>se</sub>ttin<sub>g</sub> $( w = 1 )$ <sub>,</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rd <sub>c</sub>l<sub>ass</sub> <sub>co</sub>nditi<sub>o</sub>n<sub>e</sub>d inf<sub>e</sub>r<sub>e</sub>n<sub>ce</sub> <sub>p</sub>r<sub>oce</sub>d<sub>u</sub>r<sub>e</sub> <sub>s</sub>t<sub>a</sub>rtin<sub>g</sub> <sub>w</sub>ith <sub>samp</sub>l<sub>es</sub> f<sub>rom</sub> $p _ { 0 }$ <sub>w</sub>ill <sub>no</sub>t <sub>necessar</sub>il<sub>y</sub> l<sub>an</sub>d <sub>a</sub>t $p _ { 1 } | c ,$ <sub>, a</sub>nd i<sub>s o</sub>nl<sub>y gua</sub>r<sub>a</sub>nt<sub>ee</sub>d t<sub>o</sub> if th<sub>e</sub> initi<sub>a</sub>l <sub>sa</sub>m<sub>p</sub>l<sub>es a</sub>r<sub>e</sub> dr<sub>a</sub>wn fr<sub>o</sub>m th<sub>e</sub> intr<sub>ac</sub>t<sub>a</sub>bl<sub>e</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $x _ { 0 } \mid C = c .$

![](images/dce22771c9e11656d9142a4a6d689cb42474d837c6a29466eccb14cd200022d9.jpg)  
Figure 1 2-dimensional depiction of the “yo-yo” efect. This efect pushes samples away from the center at high guidance <sub>sca</sub>l<sub>es</sub> f<sub>o</sub>r th<sub>e</sub> ind<sub>epe</sub>nd<sub>e</sub>nt <sub>a</sub>nd <sub>c</sub>l<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l OT <sub>coup</sub>lin<sub>gs,</sub> d<sub>eg</sub>r<sub>a</sub>din<sub>g ge</sub>n<sub>e</sub>r<sub>a</sub>tiv<sub>e pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce.</sub> Gl<sub>o</sub>b<sub>a</sub>l Tr<sub>a</sub>n<sub>spo</sub>rt fix<sub>es</sub> this and is stable under increasing guidance w.

## 4 Coupling Choice With Classifier-Free Guidance

The conditional source distribution mismatch described above suggests that GT may be a poor choice for un<sub>g</sub>uided conditional <sub>g</sub>eneration $( w = 1 )$ <sub>,</sub> whi<sub>c</sub>h w<sub>e o</sub>b<sub>se</sub>rv<sub>e</sub> in <sub>a</sub> 2-dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n<sub>a</sub>l <sub>4</sub>0-G<sub>auss</sub>i<sub>a</sub>n mixt<sub>u</sub>r<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l exam<sub>p</sub>le with four classes (Fi<sub>g</sub>ures 1 and 6 with details in A<sub>pp</sub>endix A). However, the result chan<sub>g</sub>es under <sub>g</sub>uidance $( w > 1 )$ <sub>.</sub> Thi<sub>s gu</sub>id<sub>a</sub>n<sub>ce</sub> d<sub>epe</sub>nd<sub>e</sub>nt <sub>e</sub>f<sub>ec</sub>t m<sub>o</sub>tiv<sub>a</sub>t<sub>es s</sub>t<sub>u</sub>d<sub>y</sub>in<sub>g</sub> th<sub>e u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>a</sub>nd <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fi<sub>e</sub>ld<sub>s</sub> together as they are used during inference, rather than judging a coupling by its unguided conditional generation <sub>p</sub>er<sup>f</sup>ormance.

## 4.1 Geometry of guided trajectories

Let $X _ { t } ^ { ( w ) }$ denote a sam<sub>p</sub>le <sub>g</sub>enerated b<sub>y</sub> inte<sub>g</sub>ratin<sub>g</sub> $\dot { X } _ { t } ^ { ( w ) } = \widetilde { u } _ { t } ^ { w } ( X _ { t } ^ { ( w ) } \mid c )$ f<sub>rom</sub> $X _ { 0 } ^ { ( w ) } \sim p _ { 0 }$ <sub>.</sub> T<sub>o</sub> tr<sub>ac</sub>k it<sub>s</sub> r<sub>a</sub>di<sub>a</sub>l <sub>pos</sub>iti<sub>o</sub>n <sub>o</sub>v<sub>e</sub>r tim<sub>e,</sub> w<sub>e</sub> m<sub>easu</sub>r<sub>e</sub>

$$
Y _ { w } ( t ) = \mathbb { E } \| X _ { t } ^ { ( w ) } \| _ { 2 } .\tag{7}
$$

Unlike the distribution of linear inter<sub>p</sub>olants used in trainin<sub>g,</sub> $Y _ { w } ( t )$ describes the radial position of trajectories <sub>p</sub>r<sub>o</sub>d<sub>uce</sub>d b<sub>y</sub> th<sub>e</sub> <sub>gu</sub>id<sub>e</sub>d <sub>sa</sub>m<sub>p</sub>l<sub>e</sub>r<sub>.</sub>

In our GMM example, trajectories trained with independent pairing contract before expanding toward their class modes (Fi<sub>g</sub>ure 2). As w increases, the re-ex<sub>p</sub>ansion becomes more <sub>p</sub>ronounced and can overshoot the tar<sub>g</sub>et regions (Figure 1). We refer to this contraction followed by re-expansion as the yo-yo efect. Class-conditional OT reduces some aspects of this behavior, while GT exhibits less pronounced contraction and overshoot over the <sub>g</sub>uidance scales we test (Fi<sub>g</sub>ures 1 and 3).

## 4.2 Coupling controls the prediction gap

R<sub>eca</sub>ll fr<sub>o</sub>m <sub>equa</sub>ti<sub>o</sub>n <sub>3</sub> th<sub>e gu</sub>id<sub>a</sub>n<sub>ce</sub> v<sub>ec</sub>t<sub>o</sub>r $g ( x _ { t } , t , c ) = u _ { t } ( x _ { t } | c ) - u _ { t } ( x _ { t } )$ m<sub>easu</sub>r<sub>es</sub> th<sub>e gap</sub> b<sub>e</sub>tw<sub>ee</sub>n <sub>u</sub>n<sub>co</sub>nditional and conditional velocit<sub>y</sub> fields. Followin<sub>g</sub> Wan<sub>g</sub> et al. (2025) we call $\| g ( x , t , c ) \| ^ { 2 }$ the prediction gap and <sub>as</sub>k h<sub>o</sub>w th<sub>e</sub> tr<sub>a</sub>inin <sub>cou</sub> lin <sub>co</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>s</sub> it<sub>s s</sub>iz<sub>e.</sub>

Let π be the joint distribution of matched source points $x _ { 0 }$ <sub>,</sub> data <sub>p</sub>oints $x _ { 1 }$ <sub>, a</sub>nd <sub>co</sub>nditi<sub>o</sub>n<sub>s</sub> $C .$ . Its $( x _ { 0 } , x _ { 1 } )$ mar<sub>g</sub>inal cou<sub>p</sub>les $p _ { 0 }$ <sub>a</sub>nd $p _ { 1 }$ . Define the trainin<sub>g</sub> inter<sub>p</sub>olant $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ and its cou<sub>p</sub>lin<sub>g</sub> cost $\operatorname { C o s t } ( \pi ) =$

![](images/a6bd71470d7ad79246254a4a2d954973907b79ff6175aec00686078d1b813771.jpg)

Figure 2 Density over time for Independent, Class conditional, and GT couplings. Independent $\mathrm { \bf { \ddot { y } 0 y 0 } } ^ { \prime \prime } \mathrm { \bf { S } , }$ contractin<sub>g</sub> towards the class ori in then ex andin outwards, whereas GT is stable.  
![](images/c6f7e8f3da194cbe68a5b8e4d6e6e2f06f9221504ccc96e3d68b5bc03c282a4a.jpg)  
Figure 3 Guided dynamics on the 40-GMM across guidance scales w. From left: state norm along guided trajectories; inte<sub>g</sub>rated <sub>g</sub>uided velocit<sub>y</sub> norm<sub>;</sub> inte<sub>g</sub>rated <sub>p</sub>rediction <sub>g</sub>a<sub>p</sub> $\| v _ { c } - v _ { \emptyset } \|$ <sub>; u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>a</sub>nd <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l m<sub>a</sub>t<sub>e</sub>ri<sub>a</sub>l <sub>acce</sub>l<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n $\| a _ { t } [ v ] \|$

$$
\mathbb { E } _ { \pi } \| x _ { 1 } - x _ { 0 } \| ^ { 2 } .
$$

Proposition 4.1 (Transport cost bounds the on-path prediction gap). Suppose $u _ { t } ( x )$ and $\boldsymbol u _ { t } ( \boldsymbol x \mid c )$ are the population squared-error minimizers for the common flow-matching target $x _ { 1 } - x _ { 0 }$ under π. Then

$$
\begin{array} { r } { \mathbb { E } _ { t , \pi } \left[ \| g ( X _ { t } , t , C ) \| ^ { 2 } \right] \leq \operatorname { C o s t } ( \pi ) - \mathcal { W } _ { 2 } ^ { 2 } ( p _ { 0 } , p _ { 1 } ) . } \end{array}\tag{8}
$$

The <sub>p</sub>roof in A<sub>pp</sub>endix B.1 first ex<sub>p</sub>resses the ex<sub>p</sub>ected <sub>p</sub>rediction <sub>g</sub>a<sub>p</sub> as the diference between the o<sub>p</sub>timal <sub>u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>a</sub>nd <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fl<sub>o</sub>w-m<sub>a</sub>t<sub>c</sub>hin r<sub>e</sub> r<sub>ess</sub>i<sub>o</sub>n <sub>e</sub>rr<sub>o</sub>r<sub>s.</sub> It th<sub>e</sub>n b<sub>ou</sub>nd<sub>s</sub> th<sub>e</sub> int<sub>e</sub> r<sub>a</sub>t<sub>e</sub>d <sub>u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub> error b<sub>y</sub> the excess <sub>q</sub>uadratic trans<sub>p</sub>ort cost. A lower-cost cou<sub>p</sub>lin<sub>g</sub> can therefore <sub>g</sub>ive a ti<sub>g</sub>hter u<sub>pp</sub>er bound on the avera<sub>g</sub>e <sub>p</sub>rediction <sub>g</sub>a<sub>p</sub> alon<sub>g</sub> its trainin<sub>g</sub> inter<sub>p</sub>olants. We note that this bound holds strictl<sub>y</sub> alon<sub>g</sub> the linear trainin<sub>g</sub> inter<sub>p</sub>olants and does not <sub>g</sub>overn the distribution of states visited durin<sub>g g</sub>uided inference $( w > 1 )$ <sub>,</sub> whi<sub>c</sub>h w<sub>e</sub> v<sub>e</sub>rif<sub>y e</sub>m<sub>p</sub>iri<sub>ca</sub>ll<sub>y.</sub> W<sub>e a</sub>l<sub>so</sub> n<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t <sub>a</sub>t th<sub>e op</sub>tim<sub>a</sub>l tr<sub>a</sub>n<sub>spo</sub>rt limit<sub>,</sub> th<sub>e</sub> b<sub>ou</sub>nd <sub>co</sub>ntr<sub>ac</sub>t<sub>s</sub> t<sub>o</sub> zero because deterministic trans<sub>p</sub>ort renders class conditionin<sub>g</sub> redundant on the trainin<sub>g</sub> su<sub>pp</sub>ort. In <sub>p</sub>ractice<sub>,</sub> finite mini-batch OT retains a non-zero <sub>p</sub>rediction <sub>g</sub>a<sub>p</sub> while re<sub>g</sub>ularizin<sub>g</sub> the velocit<sub>y</sub> fields. Further<sub>,</sub> we can <sub>s</sub>h<sub>o</sub>w <sub>a</sub> hi<sub>e</sub>r<sub>a</sub>r<sub>c</sub>h<sub>y</sub> <sub>o</sub>f <sub>coup</sub>lin<sub>g</sub> <sub>cos</sub>t<sub>s</sub> b<sub>e</sub>tw<sub>ee</sub>n dif<sub>e</sub>r<sub>e</sub>nt m<sub>e</sub>th<sub>o</sub>d<sub>s</sub> <sub>co</sub>m<sub>pa</sub>r<sub>e</sub>d in thi<sub>s</sub> <sub>pape</sub>r<sub>,</sub> whi<sub>c</sub>h w<sub>e</sub> d<sub>o</sub> in th<sub>e</sub> followin<sub>g</sub> <sub>p</sub>ro<sub>p</sub>osition.

Proposition 4.2. For all β the coupling costs satisfy

$$
\mathrm { C o s t } ( \pi _ { \mathfrak { G } \mathbb { T } } ^ { \star } ) \leq \mathrm { C o s t } ( \pi _ { C ^ { 2 } O T ( \beta ) } ^ { \star } ) \leq \mathrm { C o s t } ( \pi _ { \mathrm { i n d } } ) .\tag{9}
$$

with the lower equality achieved at $\beta = 0 .$

which establishes that the GT algorithm has a tighter upperbound on the prediction gap than $\mathrm { C } ^ { 2 } \mathrm { O T } \left( \mathrm { f o r } \beta > 0 \right)$ which has a ti<sub>g</sub>hter u<sub>pp</sub>er bound on the <sub>p</sub>rediction <sub>g</sub>a<sub>p</sub> than inde<sub>p</sub>endent cou<sub>p</sub>lin<sub>g</sub>s.

Whil<sub>e</sub> th<sub>ese</sub> r<sub>esu</sub>lt<sub>s es</sub>t<sub>a</sub>bli<sub>s</sub>h ti<sub>g</sub>ht<sub>e</sub>r <sub>uppe</sub>rb<sub>ou</sub>nd<sub>s o</sub>n th<sub>e p</sub>r<sub>e</sub>di<sub>c</sub>ti<sub>o</sub>n <sub>gap,</sub> thi<sub>s</sub> d<sub>oes</sub> n<sub>o</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>causa</sub>lit<sub>y</sub> between lower <sub>p</sub>rediction <sub>g</sub>a<sub>p</sub>s and im<sub>p</sub>roved <sub>p</sub>erformance. We use the bound to motivate the <sub>p</sub>rediction-<sub>g</sub>a<sub>p</sub> measurements in Fi<sub>g</sub>ure <sub>3,</sub> not as an ex<sub>p</sub>lanation of <sub>g</sub>eneration <sub>p</sub>erformance on its own.

## 5 Experiments

We now demonstrate the efectiveness of GT on conditional generation in several diferent settings including image generation and single-cell experiments. We first show that GT improves the performance of classifier-free guided class-conditional generation on ImageNet (Deng et al., 2009) (256 × 256) when training flow matching m<sub>o</sub>d<sub>e</sub>l<sub>s a</sub>nd di<sub>s</sub>tillin<sub>g</sub> th<sub>e</sub>m int<sub>o</sub> fl<sub>o</sub>w m<sub>aps.</sub> Th<sub>e</sub>n w<sub>e</sub> m<sub>o</sub>v<sub>e</sub> t<sub>o</sub> th<sub>e case o</sub>f <sub>co</sub>ntin<sub>uous co</sub>nditi<sub>o</sub>n<sub>e</sub>d m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> in th<sub>e</sub> t<sub>e</sub>xt-t<sub>o</sub>-im<sub>age se</sub>ttin<sub>g,</sub> b<sub>e</sub>f<sub>o</sub>r<sub>e</sub> fin<sub>a</sub>ll<sub>y</sub> d<sub>e</sub>m<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>tin<sub>g</sub> th<sub>e</sub> im<sub>pac</sub>t <sub>o</sub>n <sub>s</sub>in<sub>g</sub>l<sub>e</sub>-<sub>ce</sub>ll d<sub>a</sub>t<sub>a ge</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>gu</sub>id<sub>e</sub>d b<sub>y</sub> cell type labels across multiple single-cell datasets. We denote the minibatch version of GT-MB as GT, and the semi-discrete OT version of GT as GT + SD-OT.

Table 1 Left: FID↓ comparison against baselines. <sup>†</sup>denotes results obtained with guidance interval (Kynkäänniemi et al., 2024). Right: Curated class-conditional samples from SiT-XL/2 + GT on ImageNet-256.
<table><tr><td>Method</td><td>NFE</td><td>CFG</td><td>Params</td><td>FID ↓</td></tr><tr><td colspan="5">GANs / Normalizing Flows / Autoregressive models</td></tr><tr><td>StyleGAN-XL (Sauer et al., 2022)</td><td>1</td><td>x</td><td>166M</td><td>2.30</td></tr><tr><td>STARFlow (Gu et al., 2025)</td><td>1</td><td>x</td><td>1.4B</td><td>2.40</td></tr><tr><td>VAR-d30 (Tian et al., 2024)</td><td>20</td><td></td><td>2B</td><td>1.92</td></tr><tr><td>MAR-H/2 (Li et al., 2024)</td><td>64</td><td>V</td><td>943M</td><td>1.55</td></tr><tr><td colspan="5">Diffusion / Flow models</td></tr><tr><td>ADM (Dhariwal and Nichol, 2021)</td><td>500</td><td></td><td>554M</td><td>10.94</td></tr><tr><td>LDM (Rombach et al., 2022)</td><td>500</td><td></td><td>400M</td><td>3.60</td></tr><tr><td>RIN (Jabri et al., 2022)</td><td>1000</td><td>x</td><td>410M</td><td>3.42</td></tr><tr><td>SimDiff (Hoogeboom et al., 2023)</td><td>1024</td><td></td><td>2B</td><td>2.77</td></tr><tr><td>U-ViT-H/2 (Bao et al., 2023)</td><td>100</td><td></td><td>501M</td><td>2.29</td></tr><tr><td>DiT-XL/2 (Peebles and Xie, 2023)</td><td>500</td><td></td><td>675M</td><td>2.27</td></tr><tr><td>SiT-XL/2† (Ma et al., 2024)</td><td>500</td><td></td><td>675M</td><td>2.06</td></tr><tr><td>SiT-XL/2 + GT † (ours)</td><td>500</td><td>L</td><td>675M</td><td>1.91</td></tr></table>

![](images/a7aceeeec90a9b361bb9f32d0bedf404d2454f7be96f34b7dc8dd13480e0eb4e.jpg)

## 5.1 Class- and text- conditioned image generation on ImageNet-256

We follow the h<sub>yp</sub>er<sub>p</sub>arameter set-u<sub>p</sub> of Ma et al. (2024) to train a SiT model on conditional <sub>g</sub>eneration across B/2, L/2 and XL/2 model scales using independent, class-conditional and GT transport. We report the Fréchet Ince<sub>p</sub>tion Distance (FID) (Heusel et al., 2017) and $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ (Stein et al., 2023) metrics to measure the distributional di<sub>s</sub>t<sub>a</sub>n<sub>ce</sub> b<sub>e</sub>tw<sub>ee</sub>n th<sub>e</sub> <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d <sub>a</sub>nd r<sub>ea</sub>l di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub> W<sub>e</sub> hi<sub>g</sub>hli<sub>g</sub>ht <sub>ou</sub>r b<sub>es</sub>t <sub>co</sub>nfi<sub>gu</sub>r<sub>a</sub>ti<sub>o</sub>n in T<sub>a</sub>bl<sub>e</sub> 1 <sub>s</sub>h<sub>o</sub>win<sub>g</sub> an improvement from the standard SiT-XL/2 trained with independent couplings to ours trained with the GT strate<sub>gy,</sub> im<sub>p</sub>rovin<sub>g</sub> from 2.06 to 1.<sub>9</sub>1 in FID. This one result<sub>,</sub> however<sub>,</sub> la<sub>y</sub>s in front of a more interestin<sub>g</sub> stor<sub>y</sub>.

Namely, with GT the guidance scale w can be pushed to much larger v<sub>a</sub>l<sub>ues</sub> b<sub>e</sub>f<sub>o</sub>r<sub>e</sub> <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> b<sub>eg</sub>in<sub>s</sub> t<sub>o</sub> d<sub>eg</sub>r<sub>a</sub>d<sub>e,</sub> <sub>e</sub>n<sub>a</sub>blin<sub>g</sub> <sub>us</sub> t<sub>o</sub> <sub>pus</sub>h far more a<sub>gg</sub>ressivel<sub>y</sub> into hi<sub>g</sub>h <sub>g</sub>uidance stren<sub>g</sub>th re<sub>g</sub>imes. To illustrate<sub>,</sub> consider Fi<sub>g</sub>ure <sub>4</sub> where we com<sub>p</sub>are t<sup>h</sup>e <sub>p</sub>er<sup>f</sup>ormance at various guidance scales w along with a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the <sub>g</sub>uidance interval techni<sub>q</sub>ue (K<sub>y</sub>nkäänniemi et al., 2024). We notice that for the model trained with GT couplings we can <sub>use</sub> m<sub>a</sub>rk<sub>e</sub>dl<sub>y</sub> hi<sub>g</sub>h<sub>e</sub>r <sub>gu</sub>id<sub>a</sub>n<sub>ce</sub> <sub>sca</sub>l<sub>es</sub> th<sub>an</sub> th<sub>e mo</sub>d<sub>e</sub>l t<sub>ra</sub>i<sub>ne</sub>d <sub>w</sub>ith

Table 2 Comparison of diferent couplings across various NFEs and model sizes re<sub>p</sub>orted in FID and $\mathrm { F D } _ { \mathrm { D I N O v 2 } } .$ We swe<sub>p</sub>t for the o<sub>p</sub>timal <sub>g</sub>uidance stren<sub>g</sub>th $w ^ { \ast }$ f<sub>or</sub> <sub>eac</sub>h <sub>se</sub>ttin<sub>g a</sub>nd r<sub>epo</sub>rt th<sub>e</sub> b<sub>es</sub>t r<sub>esu</sub>lt<sub>.</sub>
<table><tr><td rowspan="2">Model</td><td rowspan="2">Coupling</td><td colspan="3">FID↓</td><td colspan="3"> $\mathrm { F D } _ { \mathrm { D I N O v 2 } } \downarrow$ </td></tr><tr><td>16</td><td>32</td><td>64</td><td>16</td><td>32</td><td>64</td></tr><tr><td rowspan="4">SiT-B/2 (130M)</td><td>Independent</td><td>5.88</td><td>5.33</td><td>5.16</td><td>137.3</td><td>132.9</td><td>132.3</td></tr><tr><td>Class-cond. OT</td><td>5.79</td><td>5.20</td><td>5.08</td><td>136.4</td><td>132.2</td><td>131.4</td></tr><tr><td> $\mathsf { G T } + \mathsf { S D - O T }$ </td><td>5.11</td><td>4.90</td><td>4.85</td><td>199.6</td><td>194.6</td><td>184.7</td></tr><tr><td>GT</td><td>5.15</td><td>4.35</td><td>4.15</td><td>123.6</td><td>118.0</td><td>116.4</td></tr><tr><td rowspan="3">SiT-L/2 (459M)</td><td>Independent</td><td>5.50</td><td>5.74</td><td>2.95</td><td>91.1</td><td>84.6</td><td>83.3</td></tr><tr><td>Class-cond. OT</td><td>5.47</td><td>5.68</td><td>2.98</td><td>102.0</td><td>99.2</td><td>95.0</td></tr><tr><td>GT</td><td>3.94</td><td>3.84</td><td>2.69</td><td>83.5</td><td>78.7</td><td>73.5</td></tr><tr><td rowspan="2">SiT-XL/2 (675M)</td><td>Independent</td><td>3.95</td><td>3.00</td><td>2.78</td><td>84.2</td><td>78.7</td><td>77.8</td></tr><tr><td>GT</td><td>3.22</td><td>2.69</td><td>2.59</td><td>72.7</td><td>69.1</td><td>68.4</td></tr></table>

inde<sub>p</sub>endent cou<sub>p</sub>lin<sub>g</sub>s<sub>;</sub> alon<sub>g</sub> with havin<sub>g</sub> an overall better minimum w.r.t. FID. Observe that interval tunin<sub>g</sub> improves all the couplings and shifts their optima to larger w, whilst GT obtains the best global optima among the coupling strategies. Notably, the gap between GT and independent couplings grows as w increases. Further, observe that semi-discrete GT which is closer to the exact OT map is the most robust at higher guidance scales, h<sub>o</sub>w<sub>e</sub>v<sub>e</sub>r d<sub>oes</sub> n<sub>o</sub>t r<sub>eac</sub>h th<sub>e o</sub>v<sub>e</sub>r<sub>a</sub>ll <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l minim<sub>a</sub> FID<sub>.</sub> T<sub>o co</sub>m<sub>p</sub>l<sub>e</sub>m<sub>e</sub>nt Fi<sub>gu</sub>r<sub>e 4</sub> w<sub>e</sub> r<sub>epo</sub>rt th<sub>e</sub> b<sub>es</sub>t r<sub>esu</sub>lt<sub>s ac</sub>r<sub>oss</sub> the diferent cou<sub>p</sub>lin<sub>g</sub> strate<sub>g</sub>ies under several model sizes, swe<sub>p</sub>t over diferent <sub>g</sub>uidance scales (<sub>p</sub>er metric) in Table 2. Observe that across all NFE and model sizes the GT couplings obtain the strongest performance <sub>y</sub>ieldin<sub>g</sub> a noticeable im<sub>p</sub>rovement. For model and trainin<sub>g</sub> confi<sub>g</sub>urations <sub>p</sub>lease refer to a<sub>pp</sub>endix C.

Distillation into a flow map. We next assess whether GT couplings can serve as a distillation strategy for few-step <sub>g</sub>enerators. Followin<sub>g</sub> Lee et al. (2026), we distill a SiT-B/2 and SiT-L/2 flow matchin<sub>g</sub> teacher into a flow ma<sub>p</sub> with usin the meanflow distillation objective (Gen et al., 2025). The teacher is retrained with either inde endent or GT couplings and the student is distilled with GT in both cases. On SiT-B/2, observe that applying GT only at distillation (i.e., from an independently pretrained teacher) matches pretraining with GT throughout, indicating that GT is efective <sub>p</sub>urel<sub>y</sub> as a distillation strate<sub>gy</sub> for stron<sub>g</sub>er few-ste<sub>p</sub> <sub>g</sub>enerators out<sub>p</sub>erformin<sub>g</sub> other cou<sub>p</sub>lin<sub>g</sub> choices.

Adaptive step sampling. We compare coupling plans under the adaptive dopri5 solver as shown in Table 4. GT <sub>a</sub>tt<sub>a</sub>in<sub>s</sub> th<sub>e</sub> l<sub>o</sub>w<sub>es</sub>t FID <sub>a</sub>nd $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ overall, while semi-discrete GT requires the fewest function evaluations and <sub>ou</sub>t<sub>pe</sub>rf<sub>o</sub>rm<sub>s</sub> th<sub>e o</sub>th<sub>e</sub>r <sub>coup</sub>lin<sub>gs a</sub>t hi<sub>g</sub>h <sub>gu</sub>id<sub>a</sub>n<sub>ce sca</sub>l<sub>es.</sub>

Continuous Conditions. We next demonstrate that GT improves text-conditioned generation under classifier-

SiT-B/2, GI [0, 0.7]  
![](images/e276244626e4e65fb725f99e535d248f2b4cfaf3edea6a4e0f1b1276ed9ae613.jpg)

Independent  
![](images/7fb66e0adf954e8c708c96657bb22cccf911f1cabd871da2660c9e7786e8e788.jpg)

Class-cond. Transport  
![](images/834ea51a2da72099e4420c42b0a400b5e3988a68563bda63b5bb37767d093066.jpg)

Global Transport  
![](images/f471781a064ad5370d015762233981d03b72c2b2bd8cf8e7e797088822a7a264.jpg)  
Figure 4 Left: Guidance scale w swee for SiT-B/2 (Euler-64 ste s) FID ↓ under a tuned uidance interval [0, 0.7]. Right: Class-conditional samples for class 339 (sorrel) from SiT-L/2 (Euler-64 steps, w = 2.5) under independent, class-conditional, and GT transport.

Table 3 Left: FID ↓ on ImageNet-256 for DMF distilled with independent, class-conditional and GT couplings, at B/2 and L/2 scale. Right: Curated sam les from GT DMF-B/2 at 4-NFE.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Coupling</td><td colspan="3">Teacher (NFE)</td><td colspan="3">Student (NFE)</td></tr><tr><td>16</td><td>32</td><td>64</td><td>1</td><td>2</td><td>4</td></tr><tr><td rowspan="4">DMF-B/2</td><td>Independent</td><td>5.88</td><td>5.33</td><td>5.16</td><td>5.82</td><td>5.19</td><td>5.19</td></tr><tr><td>Class-cond. OT</td><td>5.79</td><td>5.20</td><td>5.08</td><td>5.83</td><td>5.18</td><td>5.25</td></tr><tr><td>GT (from GT teacher)</td><td>5.15</td><td>4.35</td><td>4.15</td><td>5.66</td><td>4.17</td><td>4.14</td></tr><tr><td>GT (from ind. teacher)</td><td>5.88</td><td>5.32</td><td>5.22</td><td>5.88</td><td>4.21</td><td>4.18</td></tr><tr><td rowspan="2">DMF-L/2</td><td>Independent</td><td>5.50</td><td>5.74</td><td>2.95</td><td>3.31</td><td>2.82</td><td>2.68</td></tr><tr><td>GT</td><td>3.94</td><td>3.84</td><td>2.69</td><td>4.03</td><td>2.46</td><td>2.37</td></tr></table>

![](images/ac7920ec9b95dc74f67fc5c74d5a25655de9ddadfb3916c2918e11825cd99971.jpg)

![](images/a45a6a7ebee78c2354bad065986b28d2f3fc479cbb49d5138e0f3dda1d5150a9.jpg)

Table 4 FID ↓, $\mathrm { F D } _ { \mathrm { D I N O v 2 } } \ .$ ↓, and NFE ↓ for SiT-B/2 over cou<sub>p</sub>lin<sub>g</sub> <sub>p</sub>lans and <sub>g</sub>uidance scales w.
<table><tr><td rowspan="2">w</td><td colspan="3">Independent</td><td colspan="3">Class-cond. OT</td><td colspan="3">GT</td><td colspan="3">GT +SD-OT</td></tr><tr><td>FID</td><td> $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ </td><td>NFE</td><td>FID</td><td> $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ </td><td>NFE</td><td>FID</td><td> $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ </td><td>NFE</td><td>FID</td><td> $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ </td><td>NFE</td></tr><tr><td>1.0</td><td>25.01</td><td>586.47</td><td>50.83</td><td>25.23</td><td>589.00</td><td>47.73</td><td>29.65</td><td>618.75</td><td>49.17</td><td>42.39</td><td>772.55</td><td>38.00</td></tr><tr><td>2.0</td><td>5.17</td><td>206.41</td><td>60.99</td><td>5.08</td><td>208.96</td><td>60.56</td><td>4.06</td><td>220.08</td><td>61.54</td><td>7.56</td><td>341.00</td><td>50.00</td></tr><tr><td>3.0</td><td>11.13</td><td>138.30</td><td>76.39</td><td>10.98</td><td>137.98</td><td>75.84</td><td>8.77</td><td>130.56</td><td>73.45</td><td>4.79</td><td>226.95</td><td>62.80</td></tr><tr><td>4.0</td><td>15.77</td><td>132.94</td><td>90.04</td><td>15.56</td><td>132.06</td><td>89.86</td><td>13.37</td><td>116.34</td><td>85.63</td><td>5.47</td><td>194.00</td><td>77.09</td></tr></table>

![](images/6b4360330f525c5d1a66d6a00f132f2cf2764cdbab6a4b7ee652c342f66d45a0.jpg)

Independent  
![](images/337f6655e40bf3d202658a306ae5efca8485cd704a1518b53daf6397f1de3875.jpg)

Global Transport  
![](images/1840f1fa04fa9f76baed0149c7c9c10ceb1f7477378fe0308781351af063001d.jpg)  
“a train car with graffiti on it”

![](images/2e0ef2c13fec623830177e8604c7ca38c78240d466cafa71cd77dc434430866b.jpg)

![](images/9d1f3727c809ecd9a75f22369ea0c3a74dc5c534912e64371cec29afbb40697a.jpg)

![](images/13b7964745d230221a1bb590cf7709f7db0a512402f2513ad1d911e0751577a3.jpg)  
“a close up of a green parrot with bright red eyes”  
Figure 5 Left: FID↓ and $\mathrm { F D } _ { \mathrm { D I N O v } 2 } \downarrow$ for text-conditioned ImageNet-256 across guidance scales w. Right: Text-conditioned ImageNet samples from independent and GT coupling models at w = 4.

free <sub>g</sub>uidance. Followin<sub>g</sub> the text-conditioned Ima<sub>g</sub>eNet-256 setu<sub>p</sub> of Chen<sub>g</sub> and Schwin<sub>g</sub> (2025), we condition on ima<sub>g</sub>e ca<sub>p</sub>tions from an enriched version of Ima<sub>g</sub>eNet (VisualLa<sub>y</sub>er, 2024). Ca<sub>p</sub>tions are encoded b<sub>y</sub> a frozen pretrained CLIP text encoder followed by an MLP that maps them to the conditioning signal. The null condition used for guidance dropout is a zero vector. As shown in Figure 5, GT achieves lower FID and $\mathrm { F D } _ { \mathrm { D I N O v 2 } }$ <sub>a</sub>t hi<sub>g</sub>h<sub>e</sub>r <sub>gu</sub>id<sub>a</sub>n<sub>ce sca</sub>l<sub>es.</sub>

Table 5 Comparison of GT with single-cell generative models on distribution-matching metrics (RBF-kernel MMD and 2-Wasserstein distance), avera<sub>g</sub>ed over three seeds.
<table><tr><td></td><td colspan="2">PBMC3K</td><td colspan="2">Dentate gyrus</td><td colspan="2">HLCA</td></tr><tr><td></td><td>MMD (↓)</td><td>WD (4)</td><td>MMD (↓)</td><td>WD (↓)</td><td>MMD (↓)</td><td>WD (↓)</td></tr><tr><td>c-CFGen</td><td> $\mathbf { 0 . 4 5 \pm 0 . 0 0 }$ </td><td> ${ \bf 1 1 . 1 7 \pm 0 . 0 2 }$ </td><td> $\underline { { \mathbf { 0 . 0 6 \pm 0 . 0 0 } } }$ </td><td> $7 . 2 6 \pm \mathbf { 0 . 0 3 }$ </td><td> $\mathbf { 0 . 0 6 \pm 0 . 0 0 }$ </td><td> ${ \bf 5 . 0 6 \pm 0 . 0 0 }$ </td></tr><tr><td>c-CFGen-linear</td><td> $\underline { { \mathbf { 0 . 4 1 \pm 0 . 0 0 } } }$ </td><td> ${ \underline { { 1 0 . 2 7 } } } \pm 0 . 0 8$ </td><td> $\mathbf { 0 . 0 6 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 6 9 \pm 0 . 0 1 }$ </td><td> $\underline { { \mathbf { 0 . 0 7 \pm 0 . 0 0 } } }$ </td><td> $\mathbf { 4 . 8 9 \mathop { \pm } { = } 0 . 0 1 }$ </td></tr><tr><td>scDiffusion</td><td> $\mathbf { 0 . 6 7 \pm 0 . 0 6 }$ </td><td> ${ \bf 1 } 2 . { \bf 1 } 1 \pm { \bf 0 . 1 6 }$ </td><td> $\underline { { \mathbf { 0 . 0 6 \pm 0 . 0 0 } } }$ </td><td> ${ \bf 5 . 8 9 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 1 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 5 . 4 2 \pm 0 . 0 1 }$ </td></tr><tr><td>scVI</td><td> $\mathbf { 0 . 5 8 \pm 0 . 0 1 }$ </td><td> ${ \bf 1 3 . 3 9 \pm 0 . 1 6 }$ </td><td> $\mathbf { 0 . 1 1 \pm 0 . 0 0 }$ </td><td> ${ \bf 7 . 3 4 \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 6 . 3 9 \pm 0 . 0 1 }$ </td></tr><tr><td>GT (ours)</td><td> $\mathbf { 0 . 3 9 \pm 0 . 0 0 }$ </td><td> $\mathbf { 9 . 8 0 \ : \pm { \ : 0 . 0 2 } }$ </td><td> $\mathbf { 0 . 0 5 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $\underline { { 6 . 6 1 \pm 0 . 0 1 } }$ </td><td> $\underline { { \mathbf { 0 . 0 7 \pm 0 . 0 0 } } }$  </td><td> $\mathbf { 4 . 9 8 \pm 0 . 0 0 }$ </td></tr></table>

## 5.2 Single-cell Experiments

We evaluate GT on conditional sin<sub>g</sub>le-cell <sub>g</sub>eneration followin<sub>g</sub> Palma et al. (2025), conditionin<sub>g</sub> on cell t<sub>yp</sub>e for PBMC3K<sup>1</sup>, Dentate <sub>gy</sub>rus (La Manno et al., 2018) and HLCA (Sikkema et al., 2023). A<sub>g</sub>ainst c-CFGen (Palma et al., 2025), its linear inter<sub>p</sub>olant variant, scDifusion (Luo et al., 2024) and scVI (Ga<sub>y</sub>oso et al., 2021), GT is best on three of six metrics and second best on three out of six (Table $5 ,$ details in A<sub>pp</sub>endix D).

## 6 Related Work

Optimal Transport for Generative Models. Optimal transport (Benamou and Brenier, 2000) is widely used to im rove unconditional (Pooladian et al., 2023; Ton , Fatras, et al., 2024; Ton , Malkin, et al., 2024; Calvo-Ordonez et al., 2026) and class-conditional (Chemseddine et al., 2025; Chen<sub>g</sub> and Schwin<sub>g</sub>, 2025) <sub>g</sub>eneration, at scale via semi-discrete <sub>p</sub>otentials (Kon<sub>g</sub> et al., 2026; Mousavi-Hosseini et al., 2026) and “re-flow”-st<sub>y</sub>le strate<sub>g</sub>ies that ex<sub>p</sub>loit flow invertibilit<sub>y</sub> (Kim et al., 2025; Berthelot et al., 2026). In biolo<sub>gy</sub>, it has been a<sub>pp</sub>lied to sin<sub>g</sub>le-cell trajector<sub>y</sub> inference (Schiebin<sub>g</sub>er et al., 2019; Ka<sub>p</sub>uśniak et al., 2024; Petrović et al., 2025) and measure-to-measure trans<sub>p</sub>ort (Haviv et al., 2025; Vander<sub>g</sub>rift et al., 2026).

Flow Maps. Flow maps (Bofi et al., 2025; Frans et al., 2025; Geng et al., 2025) have recently emerged as an eficient route to one- and few-ste<sub>p</sub> <sub>g</sub>enerators, either distilled (Sabour et al., 2025; Lee et al., 2026) or trained from scratch (Bofi et al., 2025; Gen<sub>g</sub> et al., 2025, 2026), with a<sub>pp</sub>lications to ima<sub>g</sub>e (Lu, Lu, et al., 2026; Wan<sub>g</sub> et al., 2026) and video (Gu et al., 2026; Shaul et al., 2026) <sub>g</sub>eneration.

Classifier-Free Guidance for Flow Models. Classifier-free guidance (CFG) (Ho and Salimans, 2022) is a widely ado<sub>p</sub>ted strate<sub>gy</sub> for im<sub>p</sub>rovin<sub>g</sub> conditional <sub>g</sub>eneration with difusion and flow matchin<sub>g</sub> models (Zhen<sub>g</sub> et al., 2023). Subse<sub>q</sub>uent work includes <sub>g</sub>uidance interval tunin<sub>g</sub> (K<sub>y</sub>nkäänniemi et al., 2024), velocit<sub>y</sub> field <sub>p</sub>rojection (Fan et al., 2025; Cai et al., 2026), <sub>g</sub>uidance wei<sub>g</sub>ht schedules (Wan<sub>g</sub> et al., 2024; Chun<sub>g</sub> et al., 2025; Galashov et al., 2026) and <sub>g</sub>uidin<sub>g</sub> with a weaker check<sub>p</sub>oint of the same model (Karras et al., 2024).

## 7 Conclusion

In thi<sub>s</sub> w<sub>o</sub>rk<sub>,</sub> w<sub>e s</sub>t<sub>u</sub>di<sub>e</sub>d h<sub>o</sub>w <sub>coup</sub>lin<sub>g c</sub>h<sub>o</sub>i<sub>ce a</sub>f<sub>ec</sub>t<sub>s co</sub>nditi<sub>o</sub>n<sub>a</sub>l fl<sub>o</sub>w m<sub>o</sub>d<sub>e</sub>l<sub>s sa</sub>m<sub>p</sub>l<sub>e</sub>d with <sub>c</sub>l<sub>ass</sub>ifi<sub>e</sub>r-fr<sub>ee</sub> guidance. We introduced Global Transport (GT), which constructs condition-agnostic OT couplings and requires n<sub>o</sub> <sub>c</sub>h<sub>a</sub>n<sub>ge</sub> t<sub>o</sub> th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l <sub>a</sub>r<sub>c</sub>hit<sub>ec</sub>t<sub>u</sub>r<sub>e,</sub> <sub>sa</sub>m<sub>p</sub>l<sub>e</sub>r<sub>,</sub> <sub>o</sub>r <sub>gu</sub>id<sub>a</sub>n<sub>ce</sub> r<sub>u</sub>l<sub>e.</sub> Alth<sub>oug</sub>h thi<sub>s</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> m<sub>ay</sub> l<sub>ea</sub>d t<sub>o</sub> <sub>a</sub> mi<sub>s</sub>m<sub>a</sub>t<sub>c</sub>h between the source distribution seen b<sub>y</sub> each condition durin<sub>g</sub> trainin<sub>g</sub> and de<sub>g</sub>rades <sub>p</sub>erformance in the unguided conditional generation setting, in our experiments GT improves guided generation across all settings w<sub>e s</sub>t<sub>u</sub>d<sub>y.</sub> Thi<sub>s co</sub>ntr<sub>as</sub>t <sub>s</sub>h<sub>o</sub>w<sub>s</sub> th<sub>a</sub>t <sub>u</sub>n<sub>gu</sub>id<sub>e</sub>d <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> i<sub>s</sub> n<sub>o</sub>t <sub>a</sub> r<sub>e</sub>li<sub>a</sub>bl<sub>e</sub> b<sub>as</sub>i<sub>s</sub> f<sub>o</sub>r <sub>c</sub>h<sub>oos</sub>in<sub>g a coup</sub>lin<sub>g</sub> wh<sub>e</sub>n CFG is used durin<sub>g</sub> inference<sub>,</sub> and o<sub>p</sub>ens u<sub>p</sub> a new direction of in<sub>q</sub>uir<sub>y</sub> in desi<sub>g</sub>nin<sub>g</sub> cou<sub>p</sub>lin<sub>g</sub>s for more exotic inference strate<sub>g</sub>ies.

Limitations and future work Our prediction gap bound does not control the learned fields along CFG trajectories or <sub>g</sub>uarantee a resultin<sub>g</sub> hierarch<sub>y</sub> of <sub>g</sub>eneration <sub>q</sub>ualit<sub>y</sub>. Understandin<sub>g</sub> when each construction is <sub>p</sub>referable <sub>a</sub>nd h<sub>o</sub>w <sub>coup</sub>lin<sub>g c</sub>h<sub>o</sub>i<sub>ce</sub> int<sub>e</sub>r<sub>ac</sub>t<sub>s</sub> with <sub>o</sub>th<sub>e</sub>r <sub>gu</sub>id<sub>a</sub>n<sub>ce a</sub>nd <sub>pos</sub>t-tr<sub>a</sub>inin<sub>g</sub> m<sub>e</sub>th<sub>o</sub>d<sub>s a</sub>r<sub>e use</sub>f<sub>u</sub>l dir<sub>ec</sub>ti<sub>o</sub>n<sub>s</sub> f<sub>o</sub>r future work. As CFG is <sub>p</sub>rimaril<sub>y</sub> used in ima<sub>g</sub>e and video <sub>g</sub>eneration<sub>,</sub> transferrin<sub>g</sub> these <sub>g</sub>ains to domains outside <sub>o</sub>f <sub>ce</sub>ll<sub>s</sub> i<sub>n</sub> th<sub>e</sub> lif<sub>e sc</sub>i<sub>ences, w</sub>h<sub>ere o</sub>th<sub>er</sub> f<sub>ac</sub>t<sub>ors</sub> d<sub>om</sub>i<sub>na</sub>t<sub>e, rema</sub>i<sub>ns open.</sub>

## Acknowledgments

The authors would like to thank Romeo Passaro who <sub>p</sub>artici<sub>p</sub>ated in <sub>p</sub>lantin<sub>g</sub> the seeds of this idea<sub>,</sub> initia <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s a</sub>nd di<sub>scuss</sub>i<sub>o</sub>n<sub>s, as</sub> w<sub>e</sub>ll <sub>as</sub> S<sub>co</sub>tt l<sub>e</sub> R<sub>ou</sub>x f<sub>o</sub>r f<sub>ee</sub>db<sub>ac</sub>k <sub>o</sub>n initi<sub>a</sub>l dr<sub>a</sub>ft<sub>.</sub> D<sub>a</sub>n<sub>ya</sub>l R<sub>e</sub>hm<sub>a</sub>n r<sub>ece</sub>iv<sub>e</sub>d financial su<sub>pp</sub>ort from the Natural Sciences and En<sub>g</sub>ineerin<sub>g</sub> Research Council’s (NSERC) Bantin<sub>g</sub> Postdoctoral Fellowshi<sub>p</sub> under Fundin<sub>g</sub> Reference No. 1<sub>9</sub>8<sub>5</sub>06. Lazar Atanackovic was su<sub>pp</sub>orted b<sub>y</sub> the Canada CIFAR AI Chairs program. The research was enabled in part by computational resources provided by AITHYRA (https://aithyra. at), the Digital Research Alliance of Canada (https://alliancecan.ca), the Alberta Machine Intelligence Institute (https://www.amii.ca), and NVIDIA. AITHYRA is supported by the Austrian Academy of Sciences and the not-for-<sub>p</sub>rofit Boehrin<sub>g</sub>er In<sub>g</sub>elheim Stiftun<sub>g</sub>. This research is <sub>p</sub>artiall<sub>y</sub> su<sub>pp</sub>orted b<sub>y</sub> EPSRC Turin<sub>g</sub> AI World-Leadin Research Fellowshi No. EP/X0<sub>4</sub>0062/1 and EPSRC AI Hub on Mathematical Foundations of Intelli<sub>g</sub>ence: An “Erlan<sub>g</sub>en Pro<sub>g</sub>ramme” for AI No. EP/Y0288<sub>7</sub>2/1.

## References

Alber<sub>g</sub>o, Michael S., Bofi, Nicholas M., and Vanden-Eijnden, Eric (2023). “Stochastic Inter<sub>p</sub>olants: A Unif<sub>y</sub>in<sub>g</sub> Framework for Flows and Difusions”. In: arXiv preprint 2303.08797 (cit. on p. 3).

Alber<sub>g</sub>o, Michael Samuel and Vanden-Eijnden, Eric (2023). “Buildin<sub>g</sub> Normalizin<sub>g</sub> Flows with Stochastic Interpolants”. In: International Conference on Learning Representations (cit. on p. 1).

Bao, Fan, Nie, Shen, Xue, Kaiwen, Cao, Yue, Li, Chon<sub>g</sub>xuan, Su, Han<sub>g</sub>, and Zhu, Jun (2023). “All are worth words: A vit backbone for difusion models”. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, pp. 22669–22679 (cit. on p. 7).

Benamou, Jean-David and Brenier, Yann (2000). “A com<sub>p</sub>utational fluid mechanics solution to the Mon<sub>g</sub>e-Kantorovich mass transfer problem”. In: Numerische Mathematik 84.3, pp. 375–393 (cit. on pp. 10, 21).

B<sub>er</sub>th<sub>e</sub>l<sub>o</sub>t D<sub>av</sub>id Ch<sub>en</sub> Ti<sub>anron</sub> G<sub>u</sub> Ji<sub>a</sub>t<sub>ao</sub> C<sub>u</sub>t<sub>ur</sub>i M<sub>arco</sub> Di<sub>n</sub>h L<sub>auren</sub>t Ch<sub>an</sub>d<sub>na</sub> Bh<sub>av</sub>ik Kl<sub>e</sub>i<sub>n</sub> Mi<sub>c</sub>h<sub>a</sub>l Susskind, Josh, and Zhai, Shuan<sub>g</sub>fei (2026). “The cou<sub>p</sub>lin<sub>g</sub> within: Flow matchin<sub>g</sub> via distilled normalizin<sub>g</sub> flows”. In: arXiv preprint arXiv:2603.09014 (cit. on pp. 1, 10).

Blattmann, Andreas, Dockhorn, Tim, Kulal, Sumith, Mendelevitch, Daniel, Kilian, Maciej, Lorenz, Dominik, Levi, Yam, En lish, Zion, Voleti, Vikram, Letts, Adam, et al. (2023). “Stable video difusion: Scalin latent video difusion models to large datasets”. In: arXiv preprint arXiv:2311.15127 (cit. on p. 1).

Bofi, Nicholas, Alber<sub>g</sub>o, Michael, and Vanden-Eijnden, Eric (2025). “How to build a consistenc<sub>y</sub> model: Learnin<sub>g</sub> flow maps via self-distillation”. In: Advances in Neural Information Processing Systems (cit. on p. 10).

Boïté, Samuel, Delon, Julie, and Nadjahi, Kimia (2026). “Ex<sub>p</sub>ected Batch O<sub>p</sub>timal Trans<sub>p</sub>ort Plans and Consequences for Flow Matching”. In: arXiv preprint arXiv:2605.12174 (cit. on p. 20).

Cai, Jian-Fen<sub>g</sub>, Liu, Haixia, Su, Zhen<sub>gy</sub>i, and Wan<sub>g</sub>, Chao (2026). “Im<sub>p</sub>rovin<sub>g</sub> Classifier-Free Guidance of Flow Matching via Manifold Projection”. In: International Conference on Machine Learning (cit. on pp. 2, 10).

C<sub>a</sub>lv<sub>o</sub>-Ord<sub>o</sub>n<sub>e</sub>z<sub>,</sub> S<sub>e</sub>r<sub>g</sub>i<sub>o,</sub> M<sub>eu</sub>ni<sub>e</sub>r<sub>,</sub> M<sub>a</sub>tthi<sub>eu,</sub> C<sub>a</sub>rt<sub>ea,</sub> Alv<sub>a</sub>r<sub>o,</sub> R<sub>e</sub>i<sub>s</sub>in<sub>ge</sub>r<sub>,</sub> Chri<sub>s</sub>t<sub>op</sub>h<sub>,</sub> G<sub>a</sub>l<sub>,</sub> Y<sub>a</sub>rin<sub>, a</sub>nd H<sub>e</sub>rn<sub>a</sub>nd<sub>e</sub>z-Lobato, Jose Miguel (2026). “Weighted Conditional Flow Matching”. In: arXiv preprint arXiv:2507.22270 (cit. on <sub>p</sub>. 10).

Chemseddine, Jannis, Ha<sub>g</sub>emann, Paul, Steidl, Gabriele, and Wald, Christian (2025). “Conditional Wasserstein distances with applications in Bayesian OT flow matching”. In: Journal ofMachine Learning Research 26.141, <sub>pp</sub>. 1–47 (cit. on <sub>pp</sub>. 2, 4, 10).

Chen<sub>g</sub>, Ho Kei and Schwin<sub>g</sub>, Alexander (2025). “The curse of conditions: Anal<sub>y</sub>zin<sub>g</sub> and im<sub>p</sub>rovin<sub>g</sub> o<sub>p</sub>timal transport for conditional flow-based generation”. In: 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, pp. 15875–15884 (cit. on pp. 2, 4, 9, 10, 30).

Chun<sub>g</sub>, H<sub>y</sub>un<sub>g</sub>jin, Kim, Jeon<sub>g</sub>sol, Park, Geon Yeon<sub>g</sub>, Nam, H<sub>y</sub>elin, and Ye, Jon<sub>g</sub> Chul (2025). “Cf<sub>g</sub>++: Manifoldconstrained classifier free guidance for difusion models”. In: International Conference on Learning Representations (cit. on p. 10).

Cuturi, Marco (2013). “Sinkhorn distances: Lightspeed computation of optimal transport”. In: Advances in Neural Information Processing Systems (cit. on p. 4).

Den<sub>g</sub>, Jia, Don<sub>g</sub>, Wei, Socher, Richard, Li, Li-Jia, Li, Kai, and Fei-Fei, Li (2009). “Ima<sub>g</sub>enet: A lar<sub>g</sub>e-scale hierarchical image database”. In: 2009 IEEE conference on computer vision and pattern recognition. IEEE, pp. 248–255 (cit. on <sub>p</sub>. 7).

Dhariwal, Prafulla and Nichol, Alexander Quinn (2021). “Difusion Models Beat GANs on Ima<sub>g</sub>e S<sub>y</sub>nthesis”. In: Advances in Neural Information Processing Systems (cit. on p. 7).

E<sub>sser,</sub> P<sub>a</sub>t<sub>r</sub>i<sub>c</sub>k<sub>,</sub> K<sub>u</sub>l<sub>a</sub>l<sub>,</sub> S<sub>um</sub>ith<sub>,</sub> Bl<sub>a</sub>tt<sub>mann,</sub> A<sub>n</sub>d<sub>reas,</sub> E<sub>n</sub>t<sub>ezar</sub>i<sub>,</sub> R<sub>a</sub>hi<sub>m,</sub> Müll<sub>er,</sub> J<sub>onas,</sub> S<sub>a</sub>i<sub>n</sub>i<sub>,</sub> H<sub>arry,</sub> L<sub>ev</sub>i<sub>,</sub> Y<sub>am,</sub> L<sub>orenz,</sub> Dominik, Sauer, Axel, Boesel, Frederic, et al. (2024). “Scalin<sub>g</sub> rectified flow transformers for hi<sub>g</sub>h-resolution image synthesis”. In: International Conference on Machine Learning (cit. on pp. 1, 2).

Fan, Weichen, Zhen<sub>g</sub>, Amber Yijia, Yeh, Ra<sub>y</sub>mond A, and Liu, Ziwei (2025). “Cf<sub>g</sub>-zero\*: Im<sub>p</sub>roved classifier-free guidance for flow matching models”. In: arXiv preprint arXiv:2503.18886 (cit. on p. 10).

Fatras, Kilian, Zine, Younes, Majewski, Sz<sub>y</sub>mon, Flamar<sub>y</sub>, Rémi, Gribonval, Rémi, and Court<sub>y</sub>, Nicolas (2021). “Minibatch optimal transport distances; analysis and applications”. In: arXiv preprint arXiv:2101.01792 (cit. on <sub>p</sub>. 4).

Frans, Kevin, Hafner, Danijar, Levine, Ser<sub>g</sub>e<sub>y</sub>, and Abbeel, Pieter (2025). “One ste<sub>p</sub> difusion via shortcut models”. In: International Conference on Learning Representations (cit. on p. 10).

G<sub>a</sub>l<sub>as</sub>h<sub>o</sub>v<sub>,</sub> Al<sub>e</sub>x<sub>a</sub>ndr<sub>e,</sub> P<sub>o</sub>kl<sub>e,</sub> A<sub>s</sub>hwini<sub>,</sub> D<sub>ouce</sub>t<sub>,</sub> Arn<sub>au</sub>d<sub>,</sub> Gr<sub>e</sub>tt<sub>o</sub>n<sub>,</sub> Arth<sub>u</sub>r<sub>,</sub> D<sub>e</sub>lbr<sub>ac</sub>i<sub>o,</sub> M<sub>au</sub>ri<sub>c</sub>i<sub>o, a</sub>nd B<sub>o</sub>rt<sub>o</sub>li<sub>,</sub> V<sub>a</sub>l<sub>e</sub>ntin De (2026). “Learn to Guide Your Difusion Model”. In: International Conference on Learning Representations (cit. on <sub>p</sub>. 10).

Ga<sub>y</sub>oso, Adam, Steier, Zoë, Lo<sub>p</sub>ez, Romain, Re<sub>g</sub>ier, Jefre<sub>y</sub>, Nazor, Kristo<sub>p</sub>her L, Streets, Aaron, and Yosef, Ni (2021). “Joint probabilistic modeling of single-cell multi-omic data with totalVI”. In: Nature methods 18.3, <sub>pp</sub>. 272–282 (cit. on <sub>p</sub>. 10).

Gen<sub>g</sub>, Zhen<sub>gy</sub>an<sub>g</sub>, Den<sub>g</sub>, Min<sub>gy</sub>an<sub>g</sub>, Bai, Xin<sub>g</sub>jian, Kolter, Zico, and He, Kaimin<sub>g</sub> (2025). “Mean flows for one-ste<sub>p</sub> generative modeling”. In: Advances in Neural Information Processing Systems (cit. on pp. 8, 10, 29).

Gen<sub>g</sub>, Zhen<sub>gy</sub>an<sub>g</sub>, Lu, Yi<sub>y</sub>an<sub>g</sub>, Wu, Zon<sub>g</sub>ze, Shechtman, Eli, Kolter, J. Zico, and He, Kaimin<sub>g</sub> (2026). “Im<sub>p</sub>roved Mean Flows: On the Challenges of Fastforward Generative Models”. In: Conference on Computer Vision and Pattern Recognition 2026 (cit. on . 10).

Gu, Jiatao, Chen, Tianrong, Berthelot, David, Zheng, Huangjie, Wang, Yuyang, ZHANG, Ruixiang, Dinh, Laurent, Bautista, Mi<sub>g</sub>uel <sup>Á</sup>n<sub>g</sub>el, Susskind, Joshua M., and Zhai, Shuan<sub>g</sub>fei (2025). “STARFlow: Scalin<sub>g</sub> Latent Normalizing Flows for High-resolution Image Synthesis”. In: Advances in Neural Information Processing Systems (cit. on <sub>p</sub>. 7).

Gu, Yuchao, Fan<sub>g</sub>, Guian, Jian<sub>g</sub>, Yuxin, Mao, Weijia, Han, Son<sub>g</sub>, Cai, Han, and Shou, Mike Zhen<sub>g</sub> (2026). “An<sub>y</sub>flow: Any-step video difusion model with on-policy flow map distillation”. In: European Conference on Computer Vision (cit. on p. 10).

Haviv, Doron, Pooladian, Aram-Alexandre, Pe’er, Dana, and Amos, Brandon (2025). “Wasserstein Flow Matchin<sub>g</sub>: Generative Modeling Over Families of Distributions”. In: Forty-second International Conference on Machine Learning (cit. on p. 10).

Heusel, Martin, Ramsauer, Hubert, Unterthiner, Thomas, Nessler, Bernhard, and Hochreiter, Se<sub>pp</sub> (2017). “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium”. In: Advances in neural information processing systems (cit. on p. 7).

Ho, Jonathan, Jain, Ajay, and Abbeel, Pieter (2020). “Denoising difusion probabilistic models”. In: Advances in Neural Information Processing Systems (cit. on p. 1).

Ho, Jonathan and Salimans, Tim (2022). “Classifier-free difusion guidance”. In: arXiv preprint arXiv:2207.12598 (cit. on <sub>pp</sub>. 2, 3, 10).

Ho, Jonathan, Salimans, Tim, Gritsenko, Alexe<sub>y</sub>, Chan, William, Norouzi, Mohammad, and Fleet, David (2022). “Video difusion models”. In: Advances in Neural Information Processing Systems (cit. on p. 1).

Hoo<sub>g</sub>eboom, Emiel, Heek, Jonathan, and Salimans, Tim (2023). “Sim<sub>p</sub>le difusion: End-to-end difusion for hi<sub>g</sub>h resolution images”. In: International Conference on Machine Learning (cit. on p. 7).

Jabri, Allan, Fleet, David, and Chen, Tin<sub>g</sub> (2022). “Scalable ada<sub>p</sub>tive com<sub>p</sub>utation for iterative <sub>g</sub>eneration”. In: arXiv preprint arXiv:2212.11972 (cit. on p. 7).

Ka<sub>p</sub>uśniak<sub>,</sub> Kac<sub>p</sub>er<sub>,</sub> Pota<sub>p</sub>tchik<sub>,</sub> Peter<sub>,</sub> Reu<sub>,</sub> Teodora<sub>,</sub> Zhan<sub>g,</sub> Leo<sub>,</sub> Ton<sub>g,</sub> Alexander<sub>,</sub> Bronstein<sub>,</sub> Michael<sub>,</sub> Bose<sub>,</sub> Avishek J, and Di Giovanni, Francesco (2024). “Metric flow matchin<sub>g</sub> for smooth inter<sub>p</sub>olations on the data manifold”. In: Advances in Neural Information Processing Systems (cit. on p. 10).

Karras, Tero, Aittala, Miika, K<sub>y</sub>nkäänniemi, Tuomas, Lehtinen, Jaakko, Aila, Timo, and Laine, Samuli (2024). “Guiding a difusion model with a bad version of itself”. In: Advances in Neural Information Processing Systems (cit. on <sub>p</sub>. 10).

Kerri<sub>g</sub>an, Gavin, Mi<sub>g</sub>liorini, Giosue, and Sm<sub>y</sub>th, Padhraic (2024). “D<sub>y</sub>namic conditional o<sub>p</sub>timal trans<sub>p</sub>ort throu<sub>g</sub>h simulation-free flows”. In: Advances in Neural Information Processing Systems (cit. on pp. 2, 4).

Kim, Beomsu, Hsieh, Yu-Guan, Klein, Michal, Cuturi, Marco, Ye, Jong Chul, Kawar, Bahjat, and Thornton, James (2025). “Simple ReFlow: Improved Techniques for Fast Flow Models”. In: International Conference on Learning Representations (cit. on p. 10).

Kon<sub>g</sub>, Lin<sub>g</sub>kai, Tao, Molei, Liu, Yan<sub>g</sub>, Wan<sub>g</sub>, Br<sub>y</sub>an, Fu, Jinmiao, Wan<sub>g</sub>, Chien-Chih, and Liu, Huidon<sub>g</sub> (2026). “AlignFlow: Improving Flow-based Generative Models with Semi-Discrete Optimal Transport”. In: International Conference on Learning Representations (cit. on pp. 2–4, 10).

K nkäänniemi, Tuomas, Aittala, Miika, Karras, Tero, Laine, Samuli, Aila, Timo, and Lehtinen, Jaakko (2024). “A<sub>pp</sub>l<sub>y</sub>in<sub>g g</sub>uidance in a limited interval im<sub>p</sub>roves sam<sub>p</sub>le and distribution <sub>q</sub>ualit<sub>y</sub> in difusion models”. In: Advances in Neural Information Processing Systems (cit. on pp. 7, 8, 10).

L<sub>a</sub> M<sub>a</sub>nn<sub>o,</sub> Gi<sub>oe</sub>l<sub>e,</sub> S<sub>o</sub>ld<sub>a</sub>t<sub>o</sub>v<sub>,</sub> R<sub>us</sub>l<sub>a</sub>n<sub>,</sub> Z<sub>e</sub>i<sub>se</sub>l<sub>,</sub> Amit<sub>,</sub> Br<sub>au</sub>n<sub>,</sub> Em<sub>e</sub>li<sub>e,</sub> H<sub>oc</sub>h<sub>ge</sub>rn<sub>e</sub>r<sub>,</sub> H<sub>a</sub>nn<sub>a</sub>h<sub>,</sub> P<sub>e</sub>t<sub>u</sub>kh<sub>o</sub>v<sub>,</sub> Vikt<sub>o</sub>r<sub>,</sub> Lidschreiber, Katja, Kastriti, Maria E, Lönnerber<sub>g</sub>, Peter, Furlan, Alessandro, et al. (2018). “RNA velocit<sub>y</sub> of sin<sub>g</sub>le cells”. In: Nature 560.7719, pp. 494–498 (cit. on pp. 10, 31).

Lee, K<sub>y</sub>un<sub>g</sub>min, Yu, Sih<sub>y</sub>un, and Shin, Jinwoo (2026). “Decou<sub>p</sub>led MeanFlow: Turnin<sub>g</sub> Flow Models into Flow Maps for Accelerated Sampling”. In: International Conference on Learning Representations (cit. on pp. 8, 10, 29).

Li, Tianhon<sub>g</sub>, Tian, Yon<sub>g</sub>lon<sub>g</sub>, Li, He, Den<sub>g</sub>, Min<sub>gy</sub>an<sub>g</sub>, and He, Kaimin<sub>g</sub> (2024). “Autore<sub>g</sub>ressive Ima<sub>g</sub>e Generation without Vector Quantization”. In: Advances in Neural Information Processing Systems (cit. on p. 7).

Li, Zihao, Zeng, Zhichen, Lin, Xiao, Fang, Feihao, Qu, Yanru, Xu, Zhe, Liu, Zhining, Ning, Xuying, Wei, Tianxin, Liu, Ge, et al. (2026). “Flow matching meets biology and life science: a survey”. In: npj Artificial Intelligence 2.1, <sub>p</sub>. 17 (cit. on <sub>p</sub>. 1).

Li<sub>p</sub>man, Yaron, Chen, Rick<sub>y</sub> T. Q., Ben-Hamu, Heli, Nickel, Maximilian, and Le, Matt (2023). “Flow Matchin<sub>g</sub> for Generative Modeling”. In: International Conference on Learning Representations (cit. on pp. 1, 3).

Liu, Xin<sub>g</sub>chao, Gon<sub>g</sub>, Chen<sub>gy</sub>ue, and liu, <sub>q</sub>ian<sub>g</sub> (2023). “Flow Strai<sub>g</sub>ht and Fast: Learnin<sub>g</sub> to Generate and Transfer Data with Rectified Flow”. In: International Conference on Learning Representations (cit. on p. 3).

Lu, Yiyang, Lu, Susie, Sun, Qiao, Zhao, Hanhong, Jiang, Zhicheng, Wang, Xianbang, Li, Tianhong, Geng, Zhen<sub>gy</sub>an<sub>g</sub>, and He, Kaimin<sub>g</sub> (2026). “One-ste<sub>p</sub> Latent-free Ima<sub>g</sub>e Generation with Pixel Mean Flows”. In: International Conference on Machine Learning (cit. on p. 10).

Lu, Yi<sub>y</sub>an<sub>g</sub>, Sun, Qiao, Wan<sub>g</sub>, Xianban<sub>g</sub>, Jian<sub>g</sub>, Zhichen<sub>g</sub>, Zhao, Hanhon<sub>g</sub>, and He, Kaimin<sub>g</sub> (2026). “Bidirectional Normalizing Flow: From Data to Noise and Back”. In: Conference on Computer Vision and Pattern Recognition 2026 (cit. on p. 1).

Luo, Er<sub>p</sub>ai, Hao, Minshen<sub>g</sub>, Wei, Lei, and Zhan<sub>g</sub>, Xue<sub>g</sub>on<sub>g</sub> (2024). “scDifusion: conditional <sub>g</sub>eneration of hi<sub>g</sub>hquality single-cell data using difusion model”. In: Bioinformatics 40.9, btae518 (cit. on p. 10).

Ma, Nanye, Goldstein, Mark, Albergo, Michael S., Bofi, Nicholas Matthew, Vanden-Eijnden, Eric, and Xie, Saining (2024). “SiT: Ex<sub>p</sub>lorin<sub>g</sub> Flow and Difusion-based Generative Models with Scalable Inter<sub>p</sub>olant Transformers”. In: European Conference on Computer Vision (cit. on pp. 1, 7, 29).

Malnick, Shimon, Rusanovsk<sub>y</sub>, Matan, Fried, Ohad, and Avidan, Shai (2026). “O<sub>p</sub>timal Trans<sub>p</sub>ort Flow Matchin<sub>g</sub> by Design”. In: arXiv preprint arXiv:2606.04092 (cit. on p. 1).

Mid<sub>g</sub>le<sub>y,</sub> Laurence Illin<sub>g,</sub> Stim<sub>p</sub>er<sub>,</sub> Vincent<sub>,</sub> Simm<sub>,</sub> Gre<sub>g</sub>or N. C.<sub>,</sub> Schölko<sub>p</sub>f<sub>,</sub> Bernhard<sub>,</sub> and Hernández-Lobato<sub>,</sub> José Miguel (2023). “Flow Annealed Importance Sampling Bootstrap”. In: International Conference on Learning Representations (cit. on p. 17).

Morehead, Alex, Atanackovic, Lazar, Hegde, Akshata, Wang, Yanli, Boadu, Frimpong, Selvaraj, Joel, Tong, Alexander, Krishna<sub>p</sub>ri<sub>y</sub>an, Aditi, and Chen<sub>g</sub>, Jianlin (2026). “Flow matchin<sub>g</sub> for <sub>g</sub>enerative modellin<sub>g</sub> in bioinformatics and computational biology”. In: Nature Machine Intelligence, pp. 1–18 (cit. on p. 1).

Mousavi-Hosseini, Alireza, Zhan<sub>g</sub>, Ste<sub>p</sub>hen Y., Klein, Michal, and cuturi, marco (2026). “Flow Matchin<sub>g</sub> with Semidiscrete Couplings”. In: International Conference on Learning Representations (cit. on pp. 2–4, 10).

P<sub>a</sub>lm<sub>a,</sub> Al<sub>essa</sub>ndr<sub>o,</sub> Ri<sub>c</sub>ht<sub>e</sub>r<sub>,</sub> Till<sub>,</sub> Zh<sub>a</sub>n<sub>g,</sub> H<sub>a</sub>n<sub>y</sub>i<sub>,</sub> L<sub>u</sub>b<sub>e</sub>tzki<sub>,</sub> M<sub>a</sub>n<sub>ue</sub>l<sub>,</sub> T<sub>o</sub>n<sub>g,</sub> Al<sub>e</sub>x<sub>a</sub>nd<sub>e</sub>r<sub>,</sub> Ditt<sub>a</sub>di<sub>,</sub> Andr<sub>ea, a</sub>nd Th<sub>e</sub>i<sub>s,</sub> Fabian J (2025). “Multi-Modal and Multi-Attribute Generation of Sin le Cells with CFGen”. In: International Conference on Learning Representations (cit. on pp. 10, 31).

Peebles, William and Xie, Saining (2023). “Scalable Difusion Models with Transformers”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (cit. on p. 7).

Peluchetti, Stefano (2023). “Non-denoising forward-time difusions”. In: arXiv preprint arXiv:2312.14589 (cit. on <sub>pp</sub>. 1, 3).

Petrović<sub>,</sub> Katarina<sub>,</sub> Atanackovic<sub>,</sub> Lazar<sub>,</sub> Moro<sub>,</sub> Vi<sub>gg</sub>o<sub>,</sub> Ka<sub>p</sub>uśniak<sub>,</sub> Kac<sub>p</sub>er<sub>,</sub> Ce<sub>y</sub>lan<sub>,</sub> Ismail Ilkan<sub>,</sub> Bronstein<sub>,</sub> Michael M., Bose, Joe<sub>y</sub>, and Ton<sub>g</sub>, Alexander (2025). “Curl<sub>y</sub> Flow Matchin<sub>g</sub> for Learnin<sub>g</sub> Non-<sub>g</sub>radient Field D<sub>y</sub>namics”. In: Advances in Neural Information Processing Systems (cit. on p. 10).

Polyak, Adam, Zohar, Amit, Brown, Andrew, Tjandra, Andros, Sinha, Animesh, Lee, Ann, Vyas, Apoorv, Shi, Bowen, Ma, Chih-Yao, Chuan<sub>g</sub>, Chin<sub>g</sub>-Yao, et al. (2024). “Movie <sub>g</sub>en: A cast of media foundation models”. In: arXiv preprint arXiv:2410.13720 (cit. on p. 1).

Pooladian<sub>,</sub> Aram-Alexandre<sub>,</sub> Ben-Hamu<sub>,</sub> Heli<sub>,</sub> Domin<sub>g</sub>o-Enrich<sub>,</sub> Carles<sub>,</sub> Amos<sub>,</sub> Brandon<sub>,</sub> Li<sub>p</sub>man<sub>,</sub> Yaron<sub>,</sub> and Chen, Rick<sub>y</sub> T. Q. (2023). “Multisam<sub>p</sub>le Flow Matchin<sub>g</sub>: Strai<sub>g</sub>htenin<sub>g</sub> Flows with Minibatch Cou<sub>p</sub>lin<sub>g</sub>s”. In: International Conference on Machine Learning (cit. on pp. 1, 3, 4, 10).

Rombach, Robin, Blattmann, Andreas, Lorenz, Dominik, Esser, Patrick, and Ommer, Björn (2022). “Hi<sub>g</sub>h-Resolution Image Synthesis With Latent Difusion Models”. In: Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (cit. on pp. 1, 7).

Sabour, Amirmojtaba, Fidler, Sanja, and Kreis, Karsten (2025). “Align your flow: Scaling continuous-time flow map distillation”. In: Advances in Neural Information Processing Systems (cit. on p. 10).

Sauer, Axel, Schwarz, Katja, and Gei<sub>g</sub>er, Andreas (2022). “St<sub>y</sub>leGAN-XL: Scalin<sub>g</sub> St<sub>y</sub>leGAN to Lar<sub>g</sub>e Diverse Datasets”. In: arXiv preprint arXiv:2202.00273 (cit. on p. 7).

S<sub>c</sub>hi<sub>e</sub>bin<sub>ge</sub>r<sub>,</sub> G<sub>eo</sub>fr<sub>ey,</sub> Sh<sub>u,</sub> Ji<sub>a</sub>n<sub>,</sub> T<sub>a</sub>b<sub>a</sub>k<sub>a,</sub> M<sub>a</sub>r<sub>c</sub>in<sub>,</sub> Cl<sub>ea</sub>r<sub>y,</sub> Bri<sub>a</sub>n<sub>,</sub> S<sub>u</sub>br<sub>a</sub>m<sub>a</sub>ni<sub>a</sub>n<sub>,</sub> Vid<sub>ya,</sub> S<sub>o</sub>l<sub>o</sub>m<sub>o</sub>n<sub>,</sub> Ar<sub>ye</sub>h<sub>,</sub> G<sub>ou</sub>ld<sub>,</sub> Joshua<sub>,</sub> Liu<sub>,</sub> Si<sub>y</sub>an<sub>,</sub> Lin<sub>,</sub> Stacie<sub>,</sub> Berube<sub>,</sub> Peter<sub>,</sub> Lee<sub>,</sub> Lia<sub>,</sub> Chen<sub>,</sub> Jenn<sub>y,</sub> Brumbau<sub>g</sub>h<sub>,</sub> Justin<sub>,</sub> Ri<sub>g</sub>ollet<sub>,</sub> Phili<sub>pp</sub>e<sub>,</sub> Hochedlin<sub>g</sub>er, Konrad, Jaenisch, Rudolf, Re<sub>g</sub>ev, Aviv, and Lander, Eric S. (2019). “O<sub>p</sub>timal-Trans<sub>p</sub>ort Anal<sub>y</sub>sis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming”. In: Cell 176.4, 928– 943.e22 (cit. on <sub>p</sub>. 10).

Shaul, Neta, Liu, Chao, Vahdat, Arash, and Berner, Julius (2026). “Parallel Decodin<sub>g</sub> Distillation for Fast Ima<sub>g</sub>e and Video Generation”. In: arXiv preprint arXiv:2607.26004 (cit. on p. 10).

Sikkema<sub>,</sub> Lisa<sub>,</sub> Ramírez-Suáste<sub>g</sub>ui<sub>,</sub> Ciro<sub>,</sub> Strobl<sub>,</sub> Daniel C<sub>,</sub> Gillett<sub>,</sub> Tessa E<sub>,</sub> Za<sub>pp</sub>ia<sub>,</sub> Luke<sub>,</sub> Madissoon<sub>,</sub> Elo<sub>,</sub> Markov<sub>,</sub> Nikola<sub>y</sub> S, Zara<sub>g</sub>osi, Laure-Emmanuelle, Ji, Yu<sub>g</sub>e, Ansari, Meshal, et al. (2023). “An inte<sub>g</sub>rated cell atlas of the lung in health and disease”. In: Nature medicine 29.6, pp. 1563–1577 (cit. on pp. 10, 31).

Son<sub>g</sub>, Yan<sub>g</sub>, Sohl-Dickstein, Jascha, Kin<sub>g</sub>ma, Diederik P, Kumar, Abhishek, Ermon, Stefano, and Poole, Ben (2021). “Score-Based Generative Modeling through Stochastic Diferential Equations”. In: International Conference on Learning Representations (cit. on p. 1).

St<sub>e</sub>i<sub>n,</sub> G<sub>eorge,</sub> C<sub>resswe</sub>ll<sub>,</sub> J<sub>esse</sub> C<sub>.,</sub> H<sub>osse</sub>i<sub>nza</sub>d<sub>e</sub>h<sub>,</sub> R<sub>asa,</sub> S<sub>u</sub>i<sub>,</sub> Yi<sub>,</sub> R<sub>oss,</sub> B<sub>ren</sub>d<sub>an</sub> L<sub>e</sub>i<sub>g</sub>h<sub>,</sub> Vill<sub>ecroze,</sub> V<sub>a</sub>l<sub>en</sub>ti<sub>n,</sub> Li<sub>u,</sub> Zhao<sub>y</sub>an, Caterini, Anthon<sub>y</sub> L., Ta<sub>y</sub>lor, Eric, and Loaiza-Ganem, Gabriel (2023). “Ex<sub>p</sub>osin<sub>g</sub> flaws of <sub>g</sub>enerative model evaluation metrics and their unfair treatment of difusion models”. In: Advances in Neural Information Processing Systems (cit. on p. 7).

Tian, Ke<sub>y</sub>u, Jian<sub>g</sub>, Yi, Yuan, Zehuan, Pen<sub>g</sub>, Bin<sub>gy</sub>ue, and Wan<sub>g</sub>, Liwei (2024). “Visual Autore<sub>g</sub>ressive Modelin<sub>g</sub>: Scalable Image Generation via Next-Scale Prediction”. In: Advances in Neural Information Processing Systems (cit. on <sub>p</sub>. 7).

T<sub>on ,</sub> Al<sub>exan</sub>d<sub>er,</sub> F<sub>a</sub>t<sub>ras,</sub> Kili<sub>an,</sub> M<sub>a</sub>lki<sub>n,</sub> Nik<sub>o</sub>l<sub>a ,</sub> H<sub>u ue</sub>t<sub>,</sub> G<sub>u</sub>ill<sub>aume,</sub> Zh<sub>an ,</sub> Y<sub>an</sub>l<sub>e</sub>i<sub>,</sub> R<sub>ec</sub>t<sub>or-</sub>B<sub>roo</sub>k<sub>s,</sub> J<sub>arr</sub>id<sub>,</sub> W<sub>o</sub>lf<sub>,</sub> Gu<sub>y</sub>, and Ben<sub>g</sub>io, Yoshua (2024). “Im<sub>p</sub>rovin<sub>g</sub> and <sub>g</sub>eneralizin<sub>g</sub> flow-based <sub>g</sub>enerative models with minibatch optimal transport”. In: Transactions on Machine Learning Research (TMLR) (cit. on pp. 1, 3, 4, 10).

T<sub>o</sub>n<sub>g,</sub> Al<sub>e</sub>x<sub>a</sub>nd<sub>e</sub>r<sub>,</sub> M<sub>a</sub>lkin<sub>,</sub> Nik<sub>o</sub>l<sub>ay,</sub> F<sub>a</sub>tr<sub>as,</sub> Kili<sub>a</sub>n<sub>,</sub> At<sub>a</sub>n<sub>ac</sub>k<sub>o</sub>vi<sub>c,</sub> L<sub>a</sub>z<sub>a</sub>r<sub>,</sub> Zh<sub>a</sub>n<sub>g,</sub> Y<sub>a</sub>nl<sub>e</sub>i<sub>,</sub> H<sub>ugue</sub>t<sub>,</sub> G<sub>u</sub>ill<sub>au</sub>m<sub>e,</sub> W<sub>o</sub>lf<sub>,</sub> G<sub>uy,</sub> and Bengio, Yoshua (2024). “Simulation-Free Schrödinger Bridges via Score and Flow Matching”. In: AISTATS (cit. on <sub>pp</sub>. 4, 10).

Vander<sub>g</sub>rift, Matthew, White, Martha, Pol<sub>y</sub>anski<sub>y</sub>, Yur<sub>y</sub>, Ri<sub>g</sub>ollet, Phili<sub>pp</sub>e, and Atanackovic, Lazar (2026). “Measure-to-measure Regression with Transformers”. In: arXiv preprint arXiv:2605.28075 (cit. on p. 10).

VisualLayer (2024). “Imagenet-1K-VL-Enriched”. In: Hugging Face dataset. URL: https://huggingface.co/ datasets/visual-layer/imagenet-1k-vl-enriched (cit. on pp. 9, 30).

Wan<sub>g</sub>, Kaibo, Mao, Jianda, Wu, Ton<sub>g</sub>, and Xian<sub>g</sub>, Yan<sub>g</sub> (2025). “Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations”. In: Advances in Neural Information Processing Systems (cit. on pp. 2, 5).

W<sub>a</sub>n<sub>g,</sub> Xi<sub>,</sub> D<sub>u</sub>f<sub>ou</sub>r<sub>,</sub> Ni<sub>co</sub>l<sub>as,</sub> Andr<sub>eou,</sub> N<sub>e</sub>f<sub>e</sub>li<sub>,</sub> C<sub>a</sub>ni<sub>,</sub> M<sub>a</sub>ri<sub>e</sub>-P<sub>au</sub>l<sub>e,</sub> Abr<sub>e</sub>v<sub>aya,</sub> Vi<sub>c</sub>t<sub>o</sub>ri<sub>a</sub> F<sub>e</sub>rn<sub>a</sub>nd<sub>e</sub>z<sub>,</sub> Pi<sub>ca</sub>rd<sub>,</sub> D<sub>a</sub>vid<sub>,</sub> and Kalogeiton, Vicky (2024). “Analysis of Classifier-Free Guidance Weight Schedulers”. In: arXiv preprint arXiv:2404.13040 (cit. on p. 10).

Wan<sub>g</sub>, Zidon<sub>g</sub>, Zhan<sub>g</sub>, Yi<sub>y</sub>uan, Yue, Xiao<sub>y</sub>u, Yue, Xian<sub>gy</sub>u, Li, Yan<sub>gg</sub>uan<sub>g</sub>, Ou<sub>y</sub>an<sub>g</sub>, Wanli, and Bai, Lei (2026). “Transition Models: Rethinking the Generative Learning Objective”. In: Conference on Computer Vision and Pattern Recognition 2026 (cit. on p. 10).

Zhan<sub>g</sub>, Ste<sub>p</sub>hen Y., Mousavi-Hosseini, Alireza, Klein, Michal, and Cuturi, Marco (2026). “On Fittin<sub>g</sub> Flow Models with Large Sinkhorn Couplings”. In: Transactions on Machine Learning Research (cit. on p. 4).

Zhen<sub>g</sub>, Qinqin<sub>g</sub>, Le, Matt, Shaul, Neta, Lipman, Yaron, Grover, Adit<sub>y</sub>a, and Chen, Rick<sub>y</sub> T. Q. (2023). “Guided Flows for Generative Modeling and Decision Making”. In: arXiv preprint arXiv:2311.13443 (cit. on p. 10).

## Appendices

A Synthetic Example 17   
B Proofs and Derivations . 18   
B.1 Relationship between coupling and prediction gap . . . 18   
B.2 Proof of Proposition 4.2: Hierarchy of Coupling Costs . . 21   
B.3 Coupling controls flow curvature 23   
C ImageNet-256 . 29   
C.1 Implementation Details . . . . . . . . . . . . . . . . 29   
C.2 Additional results on ImageNet-256 . . . . . . . . . . . . . . . . . . . 30   
D Single-cell Datasets 31   
E Additional Background . 32   
E.1 Classifier and classifier-free guidance . . . . . . . . . . . . . . . . 32   
F GT Uncurated Samples . 35   
G Class-conditional OTUncurated Samples 36   
H Independent Coupling Uncurated Samples . 37   
I Text-conditioned ImageNet-256 Uncurated Samples . 38   
I.1 GT text-conditioned ImageNet-256 . 39   
I.2 Independent text-conditioned ImageNet-256 . 40

## A Synthetic Example

Dataset construction. We construct a 40-Gaussian-Mixture with 4 classes. Following Midgley et al., 2023, we use a two-dimensional 40-com<sub>p</sub>onent Gaussian mixture (40-GMM). All com<sub>p</sub>onents have e<sub>q</sub>ual wei<sub>g</sub>ht and share th<sub>e</sub> i<sub>so</sub>tr<sub>op</sub>i<sub>c co</sub>v<sub>a</sub>ri<sub>a</sub>n<sub>ce</sub>

$$
\Sigma = \binom { 4 0 } { 0 } \quad \textstyle { 0 } \atop { 4 0 } \biggr ) ,\tag{10}
$$

<sub>w</sub>ith <sub>means</sub> $\mu _ { i }$ d<sub>rawn</sub> <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> b<sub>ox</sub> $[ - 4 0 , 4 0 ] ^ { 2 }$ , i.e. $\mu _ { i } \sim \mathcal { U } ( - 4 0 , 4 0 ) ^ { 2 }$ <sub>,</sub> <sub>g</sub>ivin<sub>g</sub> the densit<sub>y</sub>

$$
p _ { \mathrm { g m m } } ( x ) = \frac { 1 } { 4 0 } \sum _ { i = 1 } ^ { 4 0 } \mathcal { N } ( x ; \mu _ { i } , \Sigma ) .\tag{11}
$$

Th<sub>e</sub> <sub>g</sub>r<sub>ou</sub>nd tr<sub>u</sub>th i<sub>s</sub> <sub>s</sub>h<sub>o</sub>wn in fi<sub>gu</sub>r<sub>e</sub> 6<sub>.</sub>

Generated trajectories. We provide generated trajectories across a wide range of guidance scales for Independent, Class-conditional and GT transport. We observe that $\mathrm { \ " { y o - y o } } ^ { \prime \prime }$ is present in Independent with trajectories <sub>cu</sub>rvin inw<sub>a</sub>rd<sub>s a</sub>nd <sub>ou</sub>tw<sub>a</sub>rd<sub>s c</sub>l<sub>ose</sub> t<sub>o</sub> th<sub>e co</sub>nditi<sub>o</sub>n<sub>e</sub>d <sub>c</sub>l<sub>ass.</sub>

Velocity Field Heatmaps. We compute heatmaps for unconditional and conditional velocity fields comparing dif<sub>e</sub>r<sub>e</sub>nt <sub>coup</sub>lin<sub>g p</sub>l<sub>a</sub>n<sub>s.</sub> Und<sub>e</sub>r th<sub>e</sub> ind<sub>epe</sub>nd<sub>e</sub>nt <sub>coup</sub>lin<sub>g,</sub> b<sub>o</sub>th <sub>u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>a</sub>nd <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fi<sub>e</sub>ld <sub>s</sub>t<sub>a</sub>rt <sub>as</sub> a sink located around class origin, reorganizing rapidly at t = 0.5. Under GT we observe that unconditional velocity field forms already at t = 0 and remains stable across the trajectory.

class 0 class 1

![](images/b69fbddb2f4aadeb98bdd919ae90fa274e570eeffea7997292383cb9fc102b96.jpg)

![](images/b06ab953254ac8a7e6a1661a6fbd368790b96c0fa1cf6ff05ffc51986f4d72ad.jpg)  
class 2 class 3

Figure 6 40-GMM ground truth data. Left: Mode centers by class, Right: Ground truth samples.  
w=1  
w=1.5  
w=2  
w=2.5  
w=3  
w=4  
w=10  
w=20  
![](images/e8b9c2078c455223eff1eeaf31ffe91f784f4cfcd6e8f82455aca3c565008951.jpg)  
Figure 7 Generated trajectories for Independent, Class-conditional and GT on $4 0 { \mathrm { - } } \mathrm { G M M }$

Additional samples. We provide additional generated samples to accompany figure 1 demonstrating GT does n<sub>o</sub>t dr<sub>op o</sub>f th<sub>e</sub> m<sub>o</sub>d<sub>es e</sub>v<sub>e</sub>n in <sub>cases o</sub>f <sub>e</sub>xtr<sub>e</sub>m<sub>e</sub>l<sub>y</sub> hi<sub>g</sub>h <sub>gu</sub>id<sub>a</sub>n<sub>ce sca</sub>l<sub>e</sub> $w = 2 0$

## B Proofs and Derivations

## B.1 Relationship between coupling and prediction gap

F<sub>or</sub> th<sub>e ease o</sub>f <sub>no</sub>t<sub>a</sub>ti<sub>on we w</sub>ill <sub>wr</sub>it<sub>e</sub> $u _ { t } ( x _ { t } )$ as $v _ { u }$ <sub>a</sub>nd $\boldsymbol { u } _ { t } ( \boldsymbol { x } _ { t } \mid c )$ as $v _ { c }$

Proposition B.1. For any coupling π, $\mathbb { E } _ { \boldsymbol \pi } \left[ \left\| u _ { t } ( \boldsymbol { x } _ { t } \mid \boldsymbol { c } ) - u _ { t } ( \boldsymbol { x } _ { t } ) \right\| ^ { 2 } \right] = L _ { u } ^ { \pi } ( t ) - L _ { \boldsymbol { c } } ^ { \pi } ( t ) .$

Proof. We can write the following by adding and subtracting u to $g = v _ { c } - v _ { u }$

$$
u - v _ { u } = ( u - v _ { c } ) + ( v _ { c } - v _ { u } )\tag{12}
$$

![](images/f0969df2ccffc6ca9443bf230659978be650003a931073cb653b9c9968fec699.jpg)  
Figure 8 Velocity field heat maps for conditional $v _ { c }$ <sub>a</sub>nd <sub>u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l $v _ { \emptyset }$ for Independent, Class-conditional and GT <sup>tr</sup>a<sup>n</sup>spo<sup>rt</sup>.

B<sub>y</sub> s<sub>q</sub>uarin<sub>g</sub> and takin<sub>g</sub> the norm we <sub>g</sub>et

$$
\begin{array} { r l } & { \| u - v _ { u } \| ^ { 2 } = \| u - v _ { c } \| ^ { 2 } + \| v _ { c } - v _ { u } \| ^ { 2 } } \\ & { \qquad + \ 2 \langle u - v _ { c } , v _ { c } - v _ { u } \rangle . } \end{array}\tag{13}
$$

We now take ex<sub>p</sub>ectation over $( x _ { 0 } , x _ { 1 } , c ) \sim \pi$

$$
\begin{array} { r l } & { \mathbb { E } _ { \pi } \| u - v _ { u } \| ^ { 2 } = \mathbb { E } _ { \pi } \| u - v _ { c } \| ^ { 2 } + \mathbb { E } _ { \pi } \| v _ { c } - v _ { u } \| ^ { 2 } } \\ & { \qquad + \ 2 \mathbb { E } _ { \pi } \langle u - v _ { c } , v _ { c } - v _ { u } \rangle . } \end{array}\tag{14}
$$

Th<sub>e expec</sub>t<sub>a</sub>ti<sub>on o</sub>f th<sub>e</sub> fi<sub>na</sub>l t<sub>erm</sub> i<sub>s zero:</sub>

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol { \pi } } \left[ \left. \boldsymbol { u } - \boldsymbol { v } _ { c } , \boldsymbol { v } _ { c } - \boldsymbol { v } _ { u } \right. \right] } \\ & { \mathrm { ~ = ~ } \mathbb { E } _ { \boldsymbol { \pi } } \left[ \mathbb { E } _ { \boldsymbol { \pi } } \left[ \left. \boldsymbol { u } - \boldsymbol { v } _ { c } , \boldsymbol { v } _ { c } - \boldsymbol { v } _ { u } \right. \mid \boldsymbol { x } _ { t } , c \right] \right] } \\ & { \mathrm { ~ = ~ } \mathbb { E } _ { \boldsymbol { \pi } } \left[ \left. \mathbb { E } _ { \boldsymbol { \pi } } [ \boldsymbol { u } - \boldsymbol { v } _ { c } \mid \boldsymbol { x } _ { t } , c ] , \boldsymbol { v } _ { c } - \boldsymbol { v } _ { u } \right. \right] . } \end{array}\tag{15}
$$

![](images/2583beadfaf02dc1ec585e3de4d846bf66b9a3be9ae750dc2a07de857a0e7757.jpg)  
Figure 9 Additional 40-GMM samples across a range of guidance scales.

B<sub>y</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f th<sub>e con</sub>diti<sub>ona</sub>l fi<sub>e</sub>ld<sub>,</sub>

$$
v _ { c } ( x _ { t } , t , c ) = \mathbb { E } _ { \pi } [ u \mid x _ { t } , c ] ,\tag{16}
$$

<sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub>

$$
\mathbb { E } _ { \pi } [ u - v _ { c } \mid x _ { t } , c ] = 0 .\tag{17}
$$

Hence

$$
\mathbb { E } _ { \pi } \left[ \left. u - v _ { c } , v _ { c } - v _ { u } \right. \right] = 0 .\tag{18}
$$

T<sub>a</sub>kin<sub>g e</sub>x<sub>pec</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> in <sub>equa</sub>ti<sub>o</sub>n 1<sub>3</sub> n<sub>o</sub>w <sub>g</sub>iv<sub>es</sub>

$$
\mathbb { E } _ { \pi } \| u - v _ { u } \| ^ { 2 } = \mathbb { E } _ { \pi } \| u - v _ { c } \| ^ { 2 } + \mathbb { E } _ { \pi } \| v _ { c } - v _ { u } \| ^ { 2 } .\tag{19}
$$

U<sub>s</sub>in<sub>g</sub> th<sub>e</sub> d<sub>e</sub>finiti<sub>o</sub>n<sub>s o</sub>f $L _ { u } ^ { \pi } ( t )$ <sub>a</sub>nd $L _ { c } ^ { \pi } ( t )$ <sub>,</sub> w<sub>e o</sub>bt<sub>a</sub>in

$$
\begin{array} { r } { \Big \vert \mathbb { E } _ { \pi } \| v _ { c } - v _ { u } \| ^ { 2 } = L _ { u } ^ { \pi } ( t ) - L _ { c } ^ { \pi } ( t ) \Big \vert } \end{array}\tag{20}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>proves</sub> th<sub>e resu</sub>lt<sub>.</sub>

Proposition B.2. The error term $L _ { u } ^ { \pi } ( t )$ can be upper bounded as

$$
\begin{array} { r } { \int _ { 0 } ^ { 1 } L _ { u } ^ { \pi } ( t ) \mathrm { d } t \leq c ( \pi ) - \mathcal { W } _ { 2 } ^ { 2 } ( p _ { 0 } , p _ { 1 } ) = : d ( \pi ) , } \end{array}\tag{21}
$$

where $c ( \pi )$ is the coupling cost and $d ( \pi ) \geq 0$ denotes the excess quadratic transport cost, $i . e . ,$ , the suboptimality of π relative to the population-optimal coupling (Boïté et al., 2026).

Proof. Let us consider coupling cost $c ( \pi ) = \mathbb { E } _ { \pi } \left\lceil \left\| x _ { 1 } - x _ { 0 } \right\| ^ { 2 } \right\rceil$ <sub>a</sub>nd id<sub>ea</sub>l minimiz<sub>e</sub>r <sub>o</sub>f th<sub>e u</sub>n<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fi<sub>e</sub>ld $v _ { u } ( x , t ) = \mathbb { E } _ { \pi } \left[ u \mid x = x _ { t } \right]$ <sub>, w</sub>h<sub>ere</sub> $u = x _ { 1 } - x _ { 0 }$ . We assume that the joint distribution $\pi ( x _ { 0 } , x _ { 1 } )$ ind<sub>uces a</sub> <sub>p</sub>robabilit<sub>y p</sub>ath $x _ { t } \sim p _ { t }$ . We can then ex<sub>p</sub>ress time-de<sub>p</sub>endent error $L _ { u } ^ { \pi } ( t )$ as

$$
L _ { u } ^ { \pi } ( t ) = \mathbb { E } _ { \pi } \left[ \left\| u - v _ { u } ( x _ { t } , t ) \right\| ^ { 2 } \right]\tag{22}
$$

$$
= \mathbb { E } _ { \boldsymbol \pi } \left[ \left. u \right. ^ { 2 } \right] - 2 \underbrace { \mathbb { E } _ { \boldsymbol \pi } \langle u , v _ { u } ( x _ { t } , t ) \rangle } _ { \star } + \mathbb { E } _ { \boldsymbol { x } _ { t } \sim p _ { t } } \left[ \left. v _ { u } ( x _ { t } , t ) \right. ^ { 2 } \right]\tag{23}
$$

We further express ⋆ as

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol \pi } \big [ \langle \boldsymbol u , \boldsymbol v _ { \boldsymbol u } ( \boldsymbol x _ { t } , t ) \rangle \big ] = \mathbb { E } _ { \boldsymbol { x } _ { t } \sim p _ { t } } \Big [ \Big \langle \underbrace { \mathbb { E } _ { \boldsymbol \pi } [ \boldsymbol u \mid \boldsymbol x _ { t } ] } _ { = \boldsymbol v _ { \boldsymbol u } ( \boldsymbol x _ { t } , t ) } , ~ \boldsymbol v _ { \boldsymbol u } ( \boldsymbol x _ { t } , t ) \rangle \Big ] } \\ & { \mathrm { ~ \ ~ \ } } \\ & { \mathrm { \ ~ \ } = \mathbb { E } _ { \boldsymbol { x } _ { t } \sim p _ { t } } \big [ \langle \boldsymbol v _ { \boldsymbol u } ( \boldsymbol x _ { t } , t ) , \boldsymbol v _ { \boldsymbol u } ( \boldsymbol x _ { t } , t ) \rangle \big ] } \\ & { \mathrm { \ ~ \ ~ \ } = \mathbb { E } _ { \boldsymbol { x } _ { t } \sim p _ { t } } \big [ \big \| \boldsymbol v _ { \boldsymbol u } ( \boldsymbol x _ { t } , t ) \big \| ^ { 2 } \big ] . } \end{array}\tag{24}
$$

F<sub>rom</sub> h<sub>ere</sub> it f<sub>o</sub>ll<sub>ows</sub>

$$
L _ { u } ^ { \pi } ( t ) = \underbrace { \mathbb { E } _ { \pi } \left[ \left. u \right. ^ { 2 } \right] } _ { c ( \pi ) } - \mathbb { E } _ { x _ { t } \sim p _ { t } } \left[ \left. v _ { u } ( x _ { t } , t ) \right. ^ { 2 } \right]\tag{25}
$$

or e<sub>q</sub>uivalentl<sub>y</sub>

$$
L _ { u } ^ { \pi } ( t ) = c ( \pi ) - \mathbb { E } _ { x _ { t } \sim p _ { t } } \left[ \left. v _ { u } ( x _ { t } , t ) \right. ^ { 2 } \right]\tag{26}
$$

If we integrate both sides with respect to t

$$
\int _ { 0 } ^ { 1 } L _ { u } ^ { \pi } ( t ) \mathrm { d } t = c ( \pi ) - \int _ { 0 } ^ { 1 } \mathbb { E } _ { x _ { t } \sim p _ { t } } \left[ \left. v _ { u } ( x _ { t } , t ) \right. ^ { 2 } \right] \mathrm { d } t\tag{27}
$$

Usin<sub>g</sub> Benamou and Brenier (2000) we can ex<sub>p</sub>ress

$$
\int _ { 0 } ^ { 1 } \mathbb { E } _ { x _ { t } \sim p _ { t } } \left[ \left\| v _ { u } ( x _ { t } , t ) \right\| ^ { 2 } \right] \mathrm { d } t \geq \mathcal { W } _ { 2 } ^ { 2 } ( p _ { 0 } , p _ { 1 } )\tag{28}
$$

Fr<sub>o</sub>m whi<sub>c</sub>h id<sub>e</sub>ntit<sub>y</sub> f<sub>o</sub>ll<sub>o</sub>w<sub>s</sub>

$$
\begin{array} { r } { \boxed { \int _ { 0 } ^ { 1 } L _ { u } ^ { \pi } ( t ) \mathrm { d } t \le \underbrace { c ( \pi ) - \mathcal { W } _ { 2 } ^ { 2 } ( p _ { 0 } , p _ { 1 } ) } _ { d ( \pi ) } . } } \end{array}\tag{29}
$$

## B.2 Proof of Proposition 4.2: Hierarchy of Coupling Costs

Proposition B.3 (Coupling Cost Ordering). For all $\beta \geq 0$ , the quadratic transport costs ofthe global transport coupling $\pi _ { \mathsf { G T } } ^ { \star }$ , the class-conditional optimal transport coupling $\pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star }$ , and the independent coupling $\pi _ { \mathrm { i n d } } = p _ { 0 } \otimes p _ { 1 }$ satisfy:

$$
\mathrm { C o s t } ( \pi _ { \mathfrak { G } \mathbb { T } } ^ { \star } ) \leq \mathrm { C o s t } ( \pi _ { \mathrm { C } ^ { 2 } \mathrm { O T } ( \beta ) } ^ { \star } ) \leq \mathrm { C o s t } ( \pi _ { \mathrm { i n d } } ) ,\tag{30}
$$

with the lower equality achieved when $\beta = 0$

Proof. Let $\mathcal { X } = \mathbb { R } ^ { d }$ d<sub>e</sub>n<sub>o</sub>t<sub>e</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e space a</sub>nd $\mathcal { C }$ denote the conditionin<sub>g</sub> s<sub>p</sub>ace (endowed with metric $d _ { \mathcal { C } }$ or s<sub>q</sub>uare<sup>d</sup> norm $\| c _ { 0 } - c _ { 1 } \| ^ { 2 } )$ . Let $p _ { 0 }$ b<sub>e</sub> th<sub>e sou</sub>r<sub>ce</sub> n<sub>o</sub>i<sub>se</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>o</sub>n $\mathcal { X }$ <sub>, an</sub>d l<sub>e</sub>t $p _ { 1 } ( x _ { 1 } , c _ { 1 } )$ be the joint data-<sub>co</sub>nditi<sub>o</sub>nin<sub>g</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>o</sub>n $\mathcal { X } \times \mathcal { C }$ with s<sub>p</sub>atial mar<sub>g</sub>inal $\begin{array} { r } { p _ { 1 } ( x _ { 1 } ) = \int _ { \mathcal { C } } p _ { 1 } ( x _ { 1 } , c _ { 1 } ) \mathrm { d } c _ { 1 } } \end{array}$ <sub>.</sub> R<sub>eca</sub>ll th<sub>a</sub>t th<sub>e qua</sub>d<sub>ra</sub>ti<sub>c</sub> transport cost of any coupling π with marginals $p _ { 0 }$ <sub>a</sub>nd $p _ { 1 } ( x _ { 1 } )$ i<sub>s</sub> d<sub>e</sub>fin<sub>e</sub>d b<sub>y:</sub>

$$
\begin{array} { r } { \mathrm { C o s t } ( { \boldsymbol \pi } ) : = \mathbb { E } _ { ( x _ { 0 } , x _ { 1 } ) \sim \pi } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] . } \end{array}\tag{31}
$$

Part 1: $\mathrm { C o s t } ( \pi _ { 6 \mathsf { T } } ^ { \star } ) \leq \mathrm { C o s t } ( \pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star } )$ . By definition, the global transport coupling $\pi _ { 6 \intercal } ^ { \star }$ i<sub>s</sub> th<sub>e</sub> minimiz<sub>e</sub>r <sub>o</sub>f th<sub>e</sub> unconstrained Kantorovich o<sub>p</sub>timal trans<sub>p</sub>ort <sub>p</sub>roblem between $p _ { 0 }$ <sub>a</sub>nd th<sub>e spa</sub>ti<sub>a</sub>l d<sub>a</sub>t<sub>a</sub> m<sub>a</sub>r<sub>g</sub>in<sub>a</sub>l $p _ { 1 } ( x _ { 1 } )$

$$
\pi _ { \mathsf { G T } } ^ { \star } \in \operatorname { a r g m i n } _ { \pi \in \Pi ( p _ { 0 } , p _ { 1 } ) } \mathbb { E } _ { \pi } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] .\tag{32}
$$

Therefore, for any admissible joint distribution $\pi \in \Pi ( p _ { 0 } , p _ { 1 } )$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> b<sub>y</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> i<sub>n</sub>fi<sub>mum:</sub>

$$
\mathrm { C o s t } ( \pi _ { \mathsf { G T } } ^ { \star } ) \leq \mathrm { C o s t } ( \pi ) .\tag{33}
$$

Now consider the class-conditional o<sub>p</sub>timal trans<sub>p</sub>ort cou<sub>p</sub>lin<sub>g</sub> $\pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star }$ . For an<sub>y</sub> re<sub>g</sub>ularization stren<sub>g</sub>th $\beta \geq 0 _ { : }$ $\pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star }$ is defined as the minimizer of the joint spatial and condition transport problem:

$$
\pi _ { \mathbf { C } ^ { 2 } \mathrm { O T } ( \beta ) } ^ { \star } \in \underset { \pi \in \Pi ( p _ { 0 } , p _ { 1 } ) } { \mathrm { a r g m i n } } \ \mathbb { E } _ { ( ( x _ { 0 } , c _ { 0 } ) , ( x _ { 1 } , c _ { 1 } ) ) \sim \pi } \left[ \Vert x _ { 1 } - x _ { 0 } \Vert ^ { 2 } + \beta \Vert c _ { 1 } - c _ { 0 } \Vert ^ { 2 } \right] .\tag{34}
$$

Since the s<sub>p</sub>atial mar<sub>g</sub>inals of an<sub>y</sub> candidate cou<sub>p</sub>lin<sub>g</sub> in e<sub>q</sub>uation $3 4$ <sub>a</sub>r<sub>e</sub> <sub>co</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d t<sub>o</sub> b<sub>e</sub> $p _ { 0 }$ <sub>a</sub>nd $p _ { 1 } ( x _ { 1 } )$ <sub>,</sub> th<sub>e</sub> resultin<sub>g</sub> o<sub>p</sub>timal cou<sub>p</sub>lin<sub>g</sub> $\pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star }$ is itself an admissible cou<sub>p</sub>lin<sub>g</sub> in $\Pi ( p _ { 0 } , p _ { 1 } )$

<sup>A</sup>pp<sup>l</sup>y<sup>in</sup>g <sup>th</sup>e op<sup>tim</sup>a<sup>lit</sup>y p<sup>r</sup>ope<sup>rt</sup>y o<sup>f</sup> $\pi _ { 6 \intercal } ^ { \star }$ from e<sub>q</sub>uation <sub>33</sub> directl<sub>y</sub> to $\pi = \pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star }$ <sub>y</sub>i<sub>e</sub>ld<sub>s:</sub>

$$
\mathrm { C o s t } ( \pi _ { 6 \intercal } ^ { \star } ) \leq \mathrm { C o s t } ( \pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star } ) , \quad \forall \beta \geq 0 .\tag{35}
$$

Wh<sub>e</sub>n $\beta = 0 _ { ; }$ , the penalty on the condition labels vanishes, and the objective in equation 34 reduces identically to the objective in equation 32. Thus, $\pi _ { \mathrm { C ^ { 2 } O T ( 0 ) } } ^ { \star } = \pi _ { \mathsf { G T } } ^ { \star }$ <sub>,</sub> achievin<sub>g</sub> e<sub>q</sub>ualit<sub>y</sub>.

Part 2: $\mathrm { C o s t } ( \pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star } ) \le \mathrm { C o s t } ( \pi _ { \mathrm { i n d } } )$ . Next, we show that the independent product coupling $\pi _ { \mathrm { i n d } } = p _ { 0 } \otimes p _ { 1 }$ serves as an upper bound on Cost $\big ( \pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star } \big )$ . Let $( x _ { 0 } , c _ { 0 } ) \sim p _ { 0 } \times p c$ <sub>a</sub>nd $( x _ { 1 } , c _ { 1 } ) \sim p _ { 1 }$ b<sub>e</sub> dr<sub>a</sub>wn ind<sub>epe</sub>nd<sub>en</sub>tl<sub>y, w</sub>h<sub>ere</sub> $p _ { \mathcal { C } }$ is the mar<sub>g</sub>inal condition distribution under $p _ { 1 }$ assi<sub>g</sub>ned inde<sub>p</sub>endentl<sub>y</sub> to noise <sub>p</sub>oints $x _ { 0 }$ Because $\pi _ { \mathrm { i n d } } \in \Pi ( p _ { 0 } , p _ { 1 } )$ is an admissible joint distribution, the minimality of $\pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star }$ for the joint objective in <sub>equa</sub>ti<sub>o</sub>n <sub>34</sub> im<sub>p</sub>li<sub>es:</sub>

$$
\begin{array} { r l } & { \mathbb { E } _ { \pi _ { \mathbf { C } ^ { 2 } { \operatorname { o r } ( \beta ) } } ^ { \star } } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] \leq \mathbb { E } _ { \pi _ { \mathbf { C } ^ { 2 } { \operatorname { o r } ( \beta ) } } ^ { \star } } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] + \beta \mathbb { E } _ { \pi _ { \mathbf { C } ^ { 2 } { \operatorname { o r } ( \beta ) } } ^ { \star } } \left[ \| c _ { 1 } - c _ { 0 } \| ^ { 2 } \right] } \\ & { \qquad \leq \mathbb { E } _ { \pi _ { \operatorname { i n d } } } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } + \beta \| c _ { 1 } - c _ { 0 } \| ^ { 2 } \right] . } \end{array}\tag{36}
$$

In the inde<sub>p</sub>endent cou<sub>p</sub>lin<sub>g</sub> $\pi _ { \mathrm { i n d } }$ <sub>,</sub> the conditionin<sub>g</sub> assi<sub>g</sub>nments $c _ { 0 }$ <sub>a</sub>nd $c _ { 1 }$ <sub>a</sub>r<sub>e</sub> m<sub>u</sub>t<sub>ua</sub>ll<sub>y</sub> ind<sub>epe</sub>nd<sub>e</sub>nt id<sub>e</sub>nti<sub>ca</sub>ll<sub>y</sub> di<sub>s</sub>trib<sub>u</sub>t<sub>e</sub>d <sub>sa</sub>m<sub>p</sub>l<sub>es</sub> fr<sub>o</sub>m $p _ { \mathcal { C } }$

In th<sub>e</sub> h<sub>a</sub>rd <sub>c</sub>l<sub>ass</sub>-m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> limit $( \beta \to \infty$ <sub>o</sub>r <sub>e</sub>x<sub>ac</sub>t <sub>c</sub>l<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>coup</sub>lin<sub>g</sub> wh<sub>e</sub>r<sub>e</sub> $c _ { 0 } = c _ { 1 } )$ , t<sup>h</sup>e trans<sub>p</sub>ort problem decomposes into |C| independent within-class sub-problems. For each class $c \in { \mathcal { C } }$ with <sub>p</sub>r<sub>e</sub>v<sub>a</sub>l<sub>e</sub>n<sub>ce</sub> $p ( c )$ :

$$
\pi _ { \mathbb { C } ^ { 2 } \mathrm { O T } ( \infty ) } ^ { \star } = \sum _ { c \in \mathcal { C } } p ( c ) \pi _ { c } ^ { \star } , \quad \mathrm { ~ w h e r e ~ } \pi _ { c } ^ { \star } \in \operatorname * { a r g m i n } _ { \pi \in \Pi ( p _ { 0 } , p _ { 1 } ( \cdot | c ) ) } \mathbb { E } _ { \pi } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] .\tag{37}
$$

For ever<sub>y</sub> class $c ,$ th<sub>e p</sub>r<sub>o</sub>d<sub>uc</sub>t m<sub>easu</sub>r<sub>e</sub> $\pi _ { \mathrm { i n d } , c } = p _ { 0 } \otimes p _ { 1 } ( \cdot \mid c )$ is an admissible coupling for the c-th sub-problem. B<sub>y</sub> the o<sub>p</sub>timalit<sub>y</sub> of $\pi _ { c } ^ { \star }$ over $\Pi ( p _ { 0 } , p _ { 1 } ( \cdot \mid c ) )$

$$
\begin{array} { r } { \mathbb { E } _ { \pi _ { c } ^ { \star } } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] \leq \mathbb { E } _ { \pi _ { \mathrm { i n d } , c } } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] . } \end{array}\tag{38}
$$

Takin<sub>g</sub> the ex<sub>p</sub>ectation over the class distribution $p ( c )$ <sub>y</sub>i<sub>e</sub>ld<sub>s:</sub>

$$
\begin{array} { r l } { \displaystyle \mathrm { C o s t } \big ( \pi _ { \mathrm { C ^ { 2 } O T ( \infty ) } } ^ { \star } \big ) = \displaystyle \sum _ { c \in \mathcal { C } } p ( c ) \mathbb { E } _ { \pi _ { c } ^ { \star } } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] } & { } \\ { \displaystyle \leq \sum _ { c \in \mathcal { C } } p ( c ) \mathbb { E } _ { \pi _ { \mathrm { i n d } , c } } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] } & { } \\ { \displaystyle } & { = \mathbb { E } _ { \pi _ { \mathrm { i n d } } } \left[ \| x _ { 1 } - x _ { 0 } \| ^ { 2 } \right] = \mathrm { C o s t } \big ( \pi _ { \mathrm { i n d } } \big ) . } \end{array}\tag{39}
$$

By monotonicity of the optimal transport objective under relaxation of the condition constraint, Cos $\langle \pi _ { \mathrm { C ^ { 2 } O T } ( \beta ) } ^ { \star } \rangle \leq \mathrm { C o s t } ( \pi _ { \mathrm { i n d } } )$ h<sub>o</sub>ld<sub>s</sub> f<sub>or a</sub>ll $\beta \geq 0$

Combinin<sub>g</sub> Part 1 and Part 2 <sub>g</sub>ives the com<sub>p</sub>lete chain of ine<sub>q</sub>ualities:

$$
\mathrm { C o s t } ( \pi _ { \mathfrak { G } \mathbb { T } } ^ { \star } ) \leq \mathrm { C o s t } ( \pi _ { \mathrm { C } ^ { 2 } \mathrm { O T } ( \beta ) } ^ { \star } ) \leq \mathrm { C o s t } ( \pi _ { \mathrm { i n d } } ) ,\tag{40}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>conc</sub>l<sub>u</sub>d<sub>es</sub> th<sub>e proo</sub>f<sub>.</sub>

## B.3 Coupling controls flow curvature

We next com<sub>p</sub>are the acceleration in the flow induced b<sub>y</sub> the cou<sub>p</sub>lin<sub>g</sub> in the standardized Gaussian settin<sub>g</sub>. W<sub>e use a</sub> G<sub>auss</sub>i<sub>a</sub>n l<sub>u</sub> -in m<sub>o</sub>d<sub>e</sub>l <sub>o</sub>f mini-b<sub>a</sub>t<sub>c</sub>h OT whi<sub>c</sub>h r<sub>o</sub>vid<sub>es a</sub>n <sub>a</sub>n<sub>a</sub>l ti<sub>ca</sub>ll tr<sub>ac</sub>t<sub>a</sub>bl<sub>e</sub> r<sub>o</sub>x f<sub>o</sub>r th<sub>e</sub> intractable discrete mini-batch cou<sub>p</sub>lin<sub>g</sub> used in our ex<sub>p</sub>eriments. For a velocit<sub>y</sub> field $v _ { t } ,$ <sub>,</sub> l<sub>e</sub>t $a _ { t } [ v ] ( x ) : = \partial _ { t } v _ { t } ( x ) +$ $D _ { x } v _ { t } ( x ) v _ { t } ( x )$ b<sub>e</sub> th<sub>e acce</sub>l<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>e</sub>x<sub>pe</sub>ri<sub>e</sub>n<sub>ce</sub>d b<sub>y a pa</sub>rti<sub>c</sub>l<sub>e</sub> ${ \dot { x } } _ { t } = v _ { t } ( x _ { t } )$ f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> fl<sub>ow, a</sub>l<sub>so</sub> k<sub>nown as</sub> th<sub>e</sub> material acceleration.

Proposition B.4 (Material acceleration under Gaussian couplings). Let $p _ { 0 } = p _ { 1 } = \mathcal { N } ( 0 , \mathrm { I d } _ { d } )$ , and let $v _ { t } ^ { \mathrm { i n d } } , v _ { t , n } ,$ and $v _ { t } ^ { \star }$ denote respectively the vectorfields constructed using (i) the independent coupling, (ii) mini-batch plug-in Gaussian OT coupling with batch size n, and (iii) the population OT coupling. Fixing d, as $n  \infty ,$ the following holds pointwise in (t, x):

$$
\begin{array} { r } { a _ { t } [ v ^ { \mathrm { i n d } } ] ( x ) = \frac { x } { ( ( 1 - t ) ^ { 2 } + t ^ { 2 } ) ^ { 2 } } , \qquad a _ { t } [ v _ { n } ] ( x ) = \frac { 2 \| x \| ^ { 2 } - ( d - 1 ) } { 2 n } x + o \left( \frac { 1 } { n } \right) , \quad a _ { t } [ v ^ { \star } ] ( x ) = 0 . } \end{array}\tag{41}
$$

Define the total material acceleration $\begin{array} { r } { \mathfrak { A } ( v ) ^ { 2 } : = \mathbb { E } _ { X _ { 0 } } \int _ { 0 } ^ { 1 } \lVert a _ { t } [ v ] ( X _ { t } ) \rVert ^ { 2 } } \end{array}$ dt. Then,

$$
\begin{array} { r } { \mathfrak { A } ( v ^ { \mathrm { i n d } } ) = \sqrt { \left( 2 + \frac { 3 \pi } { 4 } \right) d } , \qquad \mathfrak { A } ( v _ { n } ) = \frac { \sqrt { d ( d ^ { 2 } + 1 8 d + 4 1 ) } } { 2 n } + o ( n ^ { - 1 } ) , \qquad \mathfrak { A } ( v ^ { \star } ) = 0 . } \end{array}\tag{42}
$$

Th<sub>us</sub> b<sub>o</sub>th <sub>po</sub>intwi<sub>se a</sub>nd t<sub>o</sub>t<sub>a</sub>l m<sub>a</sub>t<sub>e</sub>ri<sub>a</sub>l <sub>acce</sub>l<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>sca</sub>l<sub>e as</sub> $O ( n ^ { - 1 } )$ f<sub>o</sub>r G<sub>auss</sub>i<sub>a</sub>n <sub>p</sub>l<sub>ug</sub>-in mini-b<sub>a</sub>t<sub>c</sub>h OT<sub>,</sub> r<sub>e</sub>m<sub>a</sub>in $O ( 1 )$ for inde<sub>p</sub>endent FM<sub>,</sub> and vanish for <sub>p</sub>o<sub>p</sub>ulation OT.

Gaussian plug-in minibatch model We study minibatch OT flow matching using an analytically tractable Gaus-<sub>s</sub>i<sub>a</sub>n m<sub>o</sub>d<sub>e</sub>l<sub>.</sub> L<sub>e</sub>t $p _ { 0 } = \mathcal { N } ( 0 , \mathrm { { I d } ) }$ <sub>a</sub>nd $p _ { 1 } = \mathcal { N } ( 0 , \Sigma )$ b<sub>e</sub> th<sub>e sou</sub>r<sub>ce a</sub>nd t<sub>a</sub>r<sub>ge</sub>t di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub> F<sub>o</sub>r ind<sub>epe</sub>nd<sub>e</sub>nt minibatches of size n from $p _ { 0 }$ <sub>a</sub>nd $p _ { 1 }$ <sub>, we</sub> fit th<sub>e source an</sub>d t<sub>arge</sub>t <sub>means an</sub>d <sub>covar</sub>i<sub>ances, coup</sub>l<sub>e</sub> th<sub>e</sub> fitt<sub>e</sub>d Gaussians b<sub>y</sub> their Gaussian Mon<sub>g</sub>e ma<sub>p,</sub> and mar<sub>g</sub>inalize the resultin<sub>g</sub> conditional flow-matchin<sub>g</sub> fields over the randoml drawn minibatches. We em hasize that this is a Gaussian plug-in model of mini-batch trans ort that we use for its anal<sub>y</sub>tical tractabilit<sub>y,</sub> since exact anal<sub>y</sub>sis of the discrete Hun<sub>g</sub>arian cou<sub>p</sub>lin<sub>g</sub> used in our <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub> w<sub>ou</sub>ld lik<sub>e</sub>l b<sub>e s</sub>i nifi<sub>ca</sub>ntl m<sub>o</sub>r<sub>e</sub> inv<sub>o</sub>lv<sub>e</sub>d<sub>.</sub>

Writin<sub>g</sub> $\widehat { p } _ { t , n } ( x )$ <sub>a</sub>nd $\widehat { v } _ { t , n } ( x )$ f<sub>o</sub>r th<sub>e pa</sub>th d<sub>e</sub>n<sub>s</sub>it<sub>y a</sub>nd v<sub>e</sub>l<sub>oc</sub>it<sub>y</sub> ind<sub>uce</sub>d b<sub>y a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>pa</sub>ir<sub>,</sub> th<sub>e</sub> m<sub>a</sub>r<sub>g</sub>in<sub>a</sub>l fl<sub>o</sub>w fi<sub>e</sub>ld l<sub>ea</sub>rn<sub>e</sub>d b<sub>y</sub> fl<sub>o</sub>w m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> i<sub>s</sub>

$$
v _ { t , n } ( x ) = \frac { \mathbb { E } [ \widehat { p } _ { t , n } ( x ) \widehat { v } _ { t , n } ( x ) ] } { \mathbb { E } [ \widehat { p } _ { t , n } ( x ) ] } .\tag{43}
$$

H<sub>e</sub>r<sub>e a</sub>nd thr<sub>oug</sub>h<sub>ou</sub>t thi<sub>s sec</sub>ti<sub>o</sub>n<sub>,</sub> th<sub>ese e</sub>x<sub>pec</sub>t<sub>a</sub>ti<sub>o</sub>n<sub>s a</sub>r<sub>e u</sub>nd<sub>e</sub>r<sub>s</sub>t<sub>oo</sub>d t<sub>o</sub> b<sub>e</sub> t<sub>a</sub>k<sub>e</sub>n <sub>o</sub>v<sub>e</sub>r th<sub>e</sub> ind<sub>epe</sub>nd<sub>e</sub>ntl<sub>y sa</sub>m<sub>p</sub>l<sub>e</sub>d source and tar<sub>g</sub>et minibatches. This <sub>p</sub>osterior densit<sub>y</sub> wei<sub>g</sub>htin<sub>g</sub> creates a nonlinear correction even thou<sub>g</sub>h <sub>every</sub> b<sub>a</sub>t<sub>c</sub>h<sub>-con</sub>diti<sub>ona</sub>l fi<sub>e</sub>ld i<sub>s a</sub>fi<sub>ne.</sub>

Since $p _ { 0 }$ is isotro<sub>p</sub>ic<sub>,</sub> we ma<sub>y</sub> assume that the tar<sub>g</sub>et has dia<sub>g</sub>onal covariance $\Sigma = \mathrm { d i a g } ( \sigma _ { 1 } ^ { 2 } , \dots , \sigma _ { d } ^ { 2 } ) , \sigma _ { i } > 0 \nonumber$

D<sub>e</sub>fi<sub>ne</sub> $m _ { i } ( t ) = ( 1 - t ) + t \sigma _ { i }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub>

$$
\theta _ { i j } ( t ) : = \frac { \sigma _ { i } \sigma _ { j } } { \sigma _ { i } + \sigma _ { j } } \left[ \frac { 2 t \sigma _ { i } \sigma _ { j } } { \sigma _ { i } + \sigma _ { j } } \big ( m _ { i } ( t ) + m _ { j } ( t ) \big ) - m _ { i } ( t ) m _ { j } ( t ) \right] ,\tag{44}
$$

$$
\kappa _ { i } ( t ) : = - \frac { \sigma _ { i } ^ { 2 } } { m _ { i } ( t ) ^ { 2 } } \sum _ { j \neq i } \frac { \sigma _ { j } ( 2 t \sigma _ { j } - m _ { j } ( t ) ) } { ( \sigma _ { i } + \sigma _ { j } ) ^ { 2 } m _ { j } ( t ) } .\tag{45}
$$

Proposition B.5 (Gaussian plug-in minibatch field). Forfixed d, positive-definite diagonal Σ as above, and fixed $( x , t )$ , the marginalized field (equation 43) satisfies, as $n  \infty$

$$
[ v _ { t , n } ( x ) ] _ { i } = \frac { \sigma _ { i } - 1 } { m _ { i } ( t ) } x _ { i } + \frac { 1 } { n } \left[ \kappa _ { i } ( t ) x _ { i } + \frac { x _ { i } } { m _ { i } ( t ) ^ { 3 } } \sum _ { j = 1 } ^ { d } \theta _ { i j } ( t ) \frac { x _ { j } ^ { 2 } } { m _ { j } ( t ) ^ { 3 } } \right] + o ( n ^ { - 1 } ) .\tag{46}
$$

Corollary B.5.1 (Standardized endpoints). $I f \Sigma = { \mathrm { I d } } ,$ , then

$$
v _ { t , n } ( x ) = \frac { 2 t - 1 } { 4 n } \left( 2 \| x \| ^ { 2 } - ( d - 1 ) \right) x + o ( n ^ { - 1 } ) ,\tag{47}
$$

$$
\partial _ { t } v _ { t , n } ( x ) = \frac { 1 } { 2 n } \left( 2 \| x \| ^ { 2 } - ( d - 1 ) \right) x + o ( n ^ { - 1 } ) .\tag{48}
$$

Proposition B.6 (Independent Gaussian flow-matching field). For independentflow matchingwith $p _ { 0 } = \mathcal { N } ( 0 , \mathrm { H } )$ and $p _ { 1 } = { \mathcal { N } } ( 0 , \mathrm { d i a g } ( \sigma _ { 1 } ^ { 2 } , \dots , \sigma _ { d } ^ { 2 } ) )$ , define $q _ { i } ( t ) = ( \bar { 1 } - t ) ^ { 2 } + t ^ { 2 } \sigma _ { i } ^ { 2 } , r _ { i } ( t ) = t \sigma _ { i } ^ { 2 } - ( 1 - t )$ . Then

$$
[ v _ { t } ^ { \mathrm { i n d } } ( x ) ] _ { i } = \frac { r _ { i } ( t ) } { q _ { i } ( t ) } x _ { i } ,\tag{49}
$$

$$
[ \partial _ { t } v _ { t } ^ { \mathrm { i n d } } ( x ) ] _ { i } = \left[ \frac { 1 + \sigma _ { i } ^ { 2 } } { q _ { i } ( t ) } - \frac { 2 r _ { i } ( t ) ^ { 2 } } { q _ { i } ( t ) ^ { 2 } } \right] x _ { i } .\tag{50}
$$

$I f { \boldsymbol { \Sigma } } = { \mathrm { I d } }$ , these expressions specialize to

$$
v _ { t } ^ { \mathrm { i n d } } ( x ) = \frac { 2 t - 1 } { ( 1 - t ) ^ { 2 } + t ^ { 2 } } x , \qquad \partial _ { t } v _ { t } ^ { \mathrm { i n d } } ( x ) = \frac { 4 t ( 1 - t ) } { ( ( 1 - t ) ^ { 2 } + t ^ { 2 } ) ^ { 2 } } x .\tag{51}
$$

The <sub>p</sub>o<sub>p</sub>ulation Gaussian OT field is $[ v _ { t } ^ { \star } ( x ) ] _ { i } = ( \sigma _ { i } - 1 ) x _ { i } / m _ { i } ( t )$ . Unlike the <sub>p</sub>lu<sub>g</sub>-in correction<sub>,</sub> the inde<sub>p</sub>endent fi<sub>e</sub>ld in Pr<sub>opos</sub>iti<sub>o</sub>n B<sub>.</sub>6 h<sub>as</sub> n<sub>o</sub> b<sub>a</sub>t<sub>c</sub>h-<sub>s</sub>iz<sub>e</sub> d<sub>epe</sub>nd<sub>e</sub>n<sub>ce.</sub> Wh<sub>e</sub>n $\Sigma = \mathrm { I d }$ <sub>,</sub> <sub>p</sub>o<sub>p</sub>ulation OT is the identit<sub>y</sub> and has zero velocit<sub>y</sub> and time derivative, whereas inde<sub>p</sub>endent FM exhibits an order-one contraction–ex<sub>p</sub>ansion (referred to in the main <sub>p</sub>a<sub>p</sub>er as the “<sub>y</sub>o-<sub>y</sub>o” efect) des<sub>p</sub>ite havin<sub>g</sub> identical end<sub>p</sub>oint distributions. Gaussian <sub>p</sub>lu<sub>g</sub>-in minibatchin<sub>g</sub> retains a residual version of this motion<sub>,</sub> but Corollar<sub>y</sub> $\mathrm { B } . 5 . 1$ 1 <sub>s</sub>h<sub>o</sub>w<sub>s</sub> th<sub>a</sub>t it<sub>s</sub> m<sub>ag</sub>nit<sub>u</sub>d<sub>e a</sub>nd E<sub>u</sub>l<sub>e</sub>ri<sub>a</sub>n tim<sub>e</sub> d<sub>e</sub>riv<sub>a</sub>tiv<sub>e</sub> d<sub>eca as</sub> $1 / n$

We next prove Proposition B.5 and its corollary; Proposition B.6 follows from the joint Gaussian conditioning <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>o</sub>n <sub>g</sub>iv<sub>e</sub>n <sub>a</sub>ft<sub>e</sub>rw<sub>a</sub>rd<sub>.</sub>

Setup. We keep d and Σ fixed as $n \to \infty$ <sub>a</sub>nd <sub>assu</sub>m<sub>e</sub> $n > d .$ <sub>.</sub> With $\varepsilon = n ^ { - 1 / 2 }$ <sub>a</sub>nd <sub>u</sub>nbi<sub>ase</sub>d <sub>sa</sub>m<sub>p</sub>l<sub>e co</sub>v<sub>a</sub>ri<sub>a</sub>n<sub>ces,</sub> write the source and tar<sub>g</sub>et <sub>p</sub>arameters as

$$
\widehat { \mu } _ { n } ^ { 0 } = \varepsilon h ,
$$

$$
\widehat { \Sigma } _ { n } ^ { 0 } = \mathrm { I d } + \varepsilon H ,\tag{52}
$$

$$
\widehat { \mu } _ { n } ^ { 1 } = \varepsilon \Sigma ^ { 1 / 2 } h ^ { \prime } ,
$$

$$
\widehat { \Sigma } _ { n } ^ { 1 } = \Sigma + \varepsilon \Sigma ^ { 1 / 2 } H ^ { \prime } \Sigma ^ { 1 / 2 } .\tag{53}
$$

Here $h , h ^ { \prime }$ <sub>a</sub>r<sub>e</sub> ind<sub>epe</sub>nd<sub>e</sub>nt <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rd G<sub>auss</sub>i<sub>a</sub>n v<sub>ec</sub>t<sub>o</sub>r<sub>s,</sub> whil<sub>e</sub> $H , H ^ { \prime }$ <sub>a</sub>r<sub>e</sub> ind<sub>e e</sub>nd<sub>e</sub>nt <sub>ce</sub>nt<sub>e</sub>r<sub>e</sub>d Wi<sub>s</sub>h<sub>a</sub>rt fl<sub>uc</sub>t<sub>ua</sub> ti<sub>o</sub>n<sub>s</sub> <sub>a</sub>nd <sub>a</sub>r<sub>e</sub> ind<sub>epe</sub>nd<sub>e</sub>nt <sub>o</sub>f th<sub>e</sub> <sub>sa</sub>m<sub>p</sub>l<sub>e</sub> m<sub>ea</sub>n<sub>s.</sub> Th<sub>e</sub>ir l<sub>ea</sub>din<sub>g</sub> <sub>seco</sub>nd m<sub>o</sub>m<sub>e</sub>nt<sub>s</sub> <sub>a</sub>r<sub>e</sub>

$$
\mathbb { E } [ H _ { i j } H _ { k l } ] = \delta _ { i k } \delta _ { j l } + \delta _ { i l } \delta _ { j k } + O ( n ^ { - 1 } ) ,\tag{54}
$$

<sub>an</sub>d lik<sub>ew</sub>i<sub>se</sub> f<sub>or</sub> $H ^ { \prime } .$ Th<sub>e</sub> $O ( n ^ { - 1 } )$ t<sub>e</sub>rm in <sub>equa</sub>ti<sub>o</sub>n <sub>54 co</sub>ntrib<sub>u</sub>t<sub>es o</sub>nl<sub>y</sub> b<sub>eyo</sub>nd th<sub>e o</sub>rd<sub>e</sub>r r<sub>e</sub>t<sub>a</sub>in<sub>e</sub>d b<sub>e</sub>l<sub>o</sub>w<sub>.</sub>

The Mon<sub>g</sub>e ma<sub>p</sub> between the two fitted Gaussians is

$$
T _ { n } ( x ) = \widehat { \mu } _ { n } ^ { 1 } + A _ { n } ( x - \widehat { \mu } _ { n } ^ { 0 } ) , \qquad A _ { n } \widehat { \Sigma } _ { n } ^ { 0 } A _ { n } = \widehat { \Sigma } _ { n } ^ { 1 } , \qquad A _ { n } \succ 0 .\tag{55}
$$

Ex<sub>p</sub>andin<sub>g</sub> $A _ { n } = \Sigma ^ { 1 / 2 } + \varepsilon K + \varepsilon ^ { 2 } L + O _ { p } ( \varepsilon ^ { 3 } )$ and matchin<sub>g</sub> <sub>p</sub>owers in the covariance identit<sub>y</sub> <sub>g</sub>ives

$$
\Sigma ^ { 1 / 2 } K + K \Sigma ^ { 1 / 2 } = \Sigma ^ { 1 / 2 } ( H ^ { \prime } - H ) \Sigma ^ { 1 / 2 } ,\tag{56}
$$

$$
\Sigma ^ { 1 / 2 } L + L \Sigma ^ { 1 / 2 } = - K ^ { 2 } - \Sigma ^ { 1 / 2 } H K - K H \Sigma ^ { 1 / 2 } .\tag{57}
$$

Since $\Sigma$ i<sub>s</sub> di<sub>ago</sub>n<sub>a</sub>l<sub>,</sub> th<sub>e</sub> fir<sub>s</sub>t <sub>equa</sub>ti<sub>o</sub>n h<sub>as</sub> th<sub>e e</sub>ntr<sub>y</sub>wi<sub>se so</sub>l<sub>u</sub>ti<sub>o</sub>n

$$
K _ { i j } = c _ { i j } ( H _ { i j } ^ { \prime } - H _ { i j } ) , \qquad c _ { i j } : = \frac { \sigma _ { i } \sigma _ { j } } { \sigma _ { i } + \sigma _ { j } } .\tag{58}
$$

F<sub>or</sub> l<sub>a</sub>t<sub>er use,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
M _ { t } = ( 1 - t ) \mathrm { I d } + t \Sigma ^ { 1 / 2 } , \qquad W _ { t } = M _ { t } ^ { - 1 } .\tag{59}
$$

Th<sub>e</sub> di<sub>ago</sub>n<sub>a</sub>l <sub>e</sub>ntri<sub>es</sub> <sub>o</sub>f $M _ { t }$ <sub>are</sub> th<sub>e</sub> <sub>sca</sub>l<sub>ars</sub> $m _ { i } ( t )$ . The <sub>p</sub>o<sub>p</sub>ulation Gaussian OT field is

$$
v _ { t } ^ { \star } ( x ) = ( \Sigma ^ { 1 / 2 } - \mathrm { I d } ) W _ { t } x , \qquad [ v _ { t } ^ { \star } ( x ) ] _ { i } = { \frac { \sigma _ { i } - 1 } { m _ { i } ( t ) } } x _ { i } .\tag{60}
$$

Batch-conditional and marginalized fields. For a fixed pair of batch fits, set

$$
X _ { t } = ( 1 - t ) X _ { 0 } + t T _ { n } ( X _ { 0 } ) , \qquad X _ { 0 } \sim { \mathcal { N } } ( { \widehat { \mu } } _ { n } ^ { 0 } , { \widehat { \Sigma } } _ { n } ^ { 0 } ) .\tag{61}
$$

Writin<sub>g</sub>

$$
B _ { t , n } = ( 1 - t ) \mathrm { I d } + t A _ { n } , \qquad { \widehat m } _ { t , n } = ( 1 - t ) { \widehat \mu } _ { n } ^ { 0 } + t { \widehat \mu } _ { n } ^ { 1 } ,\tag{62}
$$

th<sub>e</sub> <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>pa</sub>th i<sub>s</sub> G<sub>auss</sub>i<sub>a</sub>n <sub>a</sub>nd it<sub>s</sub> FM fi<sub>e</sub>ld i<sub>s</sub> th<sub>e</sub> <sub>a</sub>fin<sub>e</sub> m<sub>ap</sub>

$$
\widehat { v } _ { t , n } ( x ) = \widehat { \mu } _ { n } ^ { 1 } - \widehat { \mu } _ { n } ^ { 0 } + ( A _ { n } - \mathrm { I d } ) B _ { t , n } ^ { - 1 } ( x - \widehat { m } _ { t , n } ) .\tag{63}
$$

If

$$
\alpha _ { t } = ( 1 - t ) h + t \Sigma ^ { 1 / 2 } h ^ { \prime } , \qquad e _ { t } = \Sigma ^ { 1 / 2 } W _ { t } ( h ^ { \prime } - h ) ,\tag{64}
$$

then ex<sub>p</sub>ansion of e<sub>q</sub>uation 6<sub>3</sub> <sub>y</sub>ields

$$
\widehat v _ { t , n } ( x ) = v _ { t } ^ { \star } ( x ) + \varepsilon v _ { t } ^ { ( 1 ) } ( x ) + \varepsilon ^ { 2 } v _ { t } ^ { ( 2 ) } ( x ) + O _ { p } ( \varepsilon ^ { 3 } ) ,\tag{65}
$$

<sub>w</sub>h<sub>ere</sub>

$$
v _ { t } ^ { ( 1 ) } ( x ) = W _ { t } K W _ { t } x + e _ { t } ,\tag{66}
$$

$$
v _ { t } ^ { ( 2 ) } ( x ) = ( W _ { t } L W _ { t } - t W _ { t } K W _ { t } K W _ { t } ) x - W _ { t } K W _ { t } \alpha _ { t } .\tag{67}
$$

Th<sub>e</sub> m<sub>a</sub>r<sub>g</sub>in<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n in <sub>equa</sub>ti<sub>o</sub>n $4 3$ w<sub>e</sub>i<sub>g</sub>ht<sub>s</sub> <sub>eac</sub>h b<sub>a</sub>t<sub>c</sub>h-<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fi<sub>e</sub>ld b<sub>y</sub> $\widehat { p } _ { t , n } ( x )$ <sub>.</sub> W<sub>e</sub> th<sub>e</sub>r<sub>e</sub>f<sub>o</sub>r<sub>e</sub> <sub>e</sub>x<sub>pa</sub>nd thi<sub>s</sub> <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>pa</sub>th d<sub>e</sub>n<sub>s</sub>it<sub>y a</sub>b<sub>ou</sub>t $p _ { t } ^ { \star } = \mathcal { N } ( 0 , M _ { t } ^ { 2 } )$ .

C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> fitt<sub>e</sub>d b<sub>a</sub>t<sub>c</sub>h<sub>es,</sub> th<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> <sub>o</sub>f $X _ { t }$ h<sub>as</sub> th<sub>e</sub> <sub>e</sub>x<sub>pa</sub>n<sub>s</sub>i<sub>o</sub>n

$$
B _ { t , n } \widehat { \Sigma } _ { n } ^ { 0 } B _ { t , n } = M _ { t } ^ { 2 } + \varepsilon G _ { t } + O _ { p } ( \varepsilon ^ { 2 } ) , \qquad G _ { t } = M _ { t } H M _ { t } + t ( K M _ { t } + M _ { t } K ) .\tag{68}
$$

Consequent<sup>l</sup>y,

$$
\widehat { p } _ { t , n } ( x ) = p _ { t } ^ { \star } ( x ) \left[ 1 + \varepsilon \ell _ { t } ( x ) + O _ { p } ( \varepsilon ^ { 2 } ) \right] ,\tag{69}
$$

<sub>w</sub>ith

$$
\boldsymbol { \ell } _ { t } ( \boldsymbol { x } ) = \alpha _ { t } ^ { \mathsf { T } } \boldsymbol { W } _ { t } ^ { 2 } \boldsymbol { x } + \frac { 1 } { 2 } \boldsymbol { x } ^ { \mathsf { T } } \boldsymbol { W } _ { t } ^ { 2 } \boldsymbol { G } _ { t } \boldsymbol { W } _ { t } ^ { 2 } \boldsymbol { x } - \frac { 1 } { 2 } \operatorname { t r } ( \boldsymbol { W } _ { t } ^ { 2 } \boldsymbol { G } _ { t } ) .\tag{70}
$$

Since $\alpha _ { t }$ <sub>a</sub>nd $G _ { t }$ <sub>a</sub>r<sub>e ce</sub>nt<sub>e</sub>r<sub>e</sub>d<sub>,</sub> $\mathbb { E } [ \ell _ { t } ( x ) ] = 0 ;$ similarly, the centered fluctuations K and $e _ { t }$ <sub>g</sub><sup>i</sup>ve $\mathbb { E } [ v _ { t } ^ { ( 1 ) } ( x ) ] = 0$ Ex<sub>p</sub>andin<sub>g</sub> the densit<sub>y</sub>-wei<sub>g</sub>hted ratio e<sub>q</sub>uation <sub>43</sub> <sub>g</sub>ives

$$
v _ { t , n } ( x ) = v _ { t } ^ { \star } ( x ) + \frac { 1 } { n } \left\{ \mathbb { E } [ v _ { t } ^ { ( 2 ) } ( x ) ] + \mathbb { E } [ \ell _ { t } ( x ) v _ { t } ^ { ( 1 ) } ( x ) ] \right\} + o ( n ^ { - 1 } ) .\tag{71}
$$

Th<sub>e seco</sub>nd-<sub>o</sub>rd<sub>e</sub>r d<sub>e</sub>n<sub>s</sub>it<sub>y</sub> fl<sub>uc</sub>t<sub>ua</sub>ti<sub>o</sub>n <sub>ca</sub>n<sub>ce</sub>l<sub>s</sub> b<sub>e</sub>tw<sub>ee</sub>n th<sub>e</sub> n<sub>u</sub>m<sub>e</sub>r<sub>a</sub>t<sub>o</sub>r <sub>a</sub>nd d<sub>e</sub>n<sub>o</sub>min<sub>a</sub>t<sub>o</sub>r<sub>.</sub> Th<sub>e seco</sub>nd <sub>e</sub>x<sub>pec</sub>t<sub>a</sub>ti<sub>o</sub>n in <sub>equa</sub>ti<sub>o</sub>n $7 1$ is <sub>p</sub>recisel<sub>y</sub> the <sub>p</sub>osterior densit<sub>y</sub>-wei<sub>g</sub>htin<sub>g</sub> term omitted b<sub>y</sub> an unwei<sub>g</sub>hted batch avera<sub>g</sub>e.

Closed-form coeficient. We now evaluate the coeficient of the order- $\cdot 1 / n$ t<sub>e</sub>rm in <sub>equa</sub>ti<sub>o</sub>n $^ { 7 1 }$ . In the <sub>p</sub>osteriorwei<sub>g</sub>htin<sub>g</sub> term $\mathbb { E } [ \ell _ { t } ( x ) v _ { t } ^ { ( 1 ) } ( x ) ]$ <sub>,</sub> th<sub>e co</sub>v<sub>a</sub>ri<sub>a</sub>n<sub>ce</sub>-d<sub>epe</sub>nd<sub>e</sub>nt <sub>pa</sub>rt <sub>o</sub>f $\ell _ { t }$ i<sub>s</sub> lin<sub>ea</sub>r in $G _ { t }$ <sub>,</sub> whil<sub>e</sub> th<sub>e</sub> m<sub>a</sub>trix-d<sub>epe</sub>nd<sub>e</sub>nt <sub>p</sub>art o<sup>f</sup> $\bar { v _ { t } ^ { ( 1 ) } }$ i<sub>s</sub> lin<sub>ea</sub>r in $K$ . Thus<sub>,</sub> usin<sub>g</sub> e<sub>q</sub>uation <sub>5</sub>8 and the Wishart moments in e<sub>q</sub>uation $5 4 \AA$ their re<sub>q</sub>uired <sub>co</sub>v<sub>a</sub>ri<sub>a</sub>n<sub>ce co</sub>ntr<sub>ac</sub>ti<sub>o</sub>n i<sub>s</sub>

$$
\mathbb { E } [ [ G _ { t } ] _ { i j } K _ { k l } ] = \theta _ { i j } ( t ) ( \delta _ { i k } \delta _ { j l } + \delta _ { i l } \delta _ { j k } ) + O ( n ^ { - 1 } ) .\tag{72}
$$

Combinin<sub>g</sub> this contraction with the second S<sub>y</sub>lvester e<sub>q</sub>uation in e<sub>q</sub>uation $5 7 ,$ <sub>w</sub>hi<sub>c</sub>h d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> th<sub>e</sub> <sub>mean</sub> <sub>o</sub>f $v _ { t } ^ { ( 2 ) }$ , <sub>g</sub><sup>i</sup>ves

$$
\mathbb { E } [ v _ { t } ^ { ( 2 ) } ( x ) ] _ { i } = \beta _ { i } ( t ) x _ { i } + O ( n ^ { - 1 } ) ,\tag{73}
$$

$$
\mathbb { E } [ \ell _ { t } ( x ) v _ { t } ^ { ( 1 ) } ( x ) ] _ { i } = \eta _ { i } ( t ) x _ { i } + \frac { x _ { i } } { m _ { i } ( t ) ^ { 3 } } \sum _ { j = 1 } ^ { d } \theta _ { i j } ( t ) \frac { x _ { j } ^ { 2 } } { m _ { j } ( t ) ^ { 3 } } - \frac { \theta _ { i i } ( t ) } { m _ { i } ( t ) ^ { 4 } } x _ { i } + O ( n ^ { - 1 } ) ,\tag{74}
$$

<sub>w</sub>h<sub>ere</sub>

$$
\beta _ { i } ( t ) = - \frac { \sigma _ { i } ^ { 2 } } { m _ { i } ( t ) ^ { 2 } } \sum _ { j = 1 } ^ { d } ( 1 + \delta _ { i j } ) \frac { \sigma _ { j } ( 2 t \sigma _ { j } - m _ { j } ( t ) ) } { ( \sigma _ { i } + \sigma _ { j } ) ^ { 2 } m _ { j } ( t ) } ,\tag{75}
$$

$$
\eta _ { i } ( t ) = \frac { \sigma _ { i } ( 2 t \sigma _ { i } - m _ { i } ( t ) ) } { m _ { i } ( t ) ^ { 3 } } .\tag{76}
$$

The dia<sub>g</sub>onal identities

$$
\beta _ { i } ( t ) = \kappa _ { i } ( t ) - \frac { 1 } { 2 } \eta _ { i } ( t ) , \qquad \frac { \theta _ { i i } ( t ) } { m _ { i } ( t ) ^ { 4 } } = \frac { 1 } { 2 } \eta _ { i } ( t )\tag{77}
$$

cancel all dia<sub>g</sub>onal linear contributions. Insertin<sub>g</sub> the remainin<sub>g</sub> terms into e<sub>q</sub>uation <sub>7</sub>1 <sub>p</sub>roves e<sub>q</sub>uation <sub>4</sub>6.

Standardized specialization. When $\Sigma = \mathrm { I d }$ <sub>, we</sub> h<sub>ave</sub> $\sigma _ { i } = m _ { i } ( t ) = 1$ <sub>a</sub>nd $2 t \sigma _ { i } - m _ { i } ( t ) = 2 t - 1$ . Equations e<sub>q</sub>uation <sub>44</sub> and e<sub>q</sub>uation <sub>45</sub> reduce to

$$
\theta _ { i j } ( t ) = \frac { 2 t - 1 } { 2 } , \qquad \kappa _ { i } ( t ) = - \frac { ( d - 1 ) ( 2 t - 1 ) } { 4 } .\tag{78}
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>tin<sub>g</sub> th<sub>ese e</sub>x<sub>p</sub>r<sub>ess</sub>i<sub>o</sub>n<sub>s</sub> int<sub>o equa</sub>ti<sub>o</sub>n $4 6$ <sub>g</sub>ives equation $4 7 ;$ diferentiating at fixed x gives equation 48, <sub>p</sub>rovin<sub>g</sub> Corollar<sub>y</sub> B.<sub>5</sub>.1. $\mathrm { A t } t = 1 / 2$ the field is in fact exactly zero for every n: exchanging the i.i.d. fitted batches <sub>p</sub>r<sub>ese</sub>rv<sub>es</sub> th<sub>e</sub> mid<sub>po</sub>int <sub>a</sub>nd r<sub>e</sub>v<sub>e</sub>r<sub>ses</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>ace</sub>m<sub>e</sub>nt<sub>.</sub>

In one dimension<sub>,</sub> the of-dia<sub>g</sub>onal sum in e<sub>q</sub>uation <sub>45</sub> is em<sub>p</sub>t<sub>y</sub>. Writin<sub>g</sub> $m ( t ) = ( 1 - t ) + t \sigma$ <sub>,</sub> th<sub>e</sub> <sub>genera</sub>l <sub>resu</sub>lt b<sub>eco</sub>m<sub>es</sub>

$$
v _ { t , n } ( x ) = \frac { \sigma - 1 } { m ( t ) } x + \frac { 1 } { n } \frac { \sigma ( 2 t \sigma - m ( t ) ) } { 2 m ( t ) ^ { 5 } } x ^ { 3 } + o ( n ^ { - 1 } ) .\tag{79}
$$

H<sub>e</sub>n<sub>ce</sub> th<sub>e e</sub>ntir<sub>e</sub> lin<sub>ea</sub>r <sub>o</sub>rd<sub>e</sub>r- $1 / n$ <sub>co</sub>rr<sub>ec</sub>ti<sub>o</sub>n <sub>ca</sub>n<sub>ce</sub>l<sub>s</sub> in <sub>o</sub>n<sub>e</sub> dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n<sub>.</sub>

Comparison with the independent coupling.

Proof of Proposition B.6. For independent $X _ { 0 } \sim \mathcal { N } ( 0 , \mathrm { I d } )$ <sub>a</sub>nd $X _ { 1 } \sim { \mathcal { N } } ( 0 , \Sigma )$ <sub>,</sub> l<sub>e</sub>t $X _ { t } = ( 1 - t ) X _ { 0 } + t X _ { 1 }$ <sub>a</sub>nd $U = X _ { 1 } - X _ { 0 }$ . Joint Gaussian conditionin<sub>g y</sub>ields

$$
v _ { t } ^ { \mathrm { i n d } } ( x ) = \mathbb { E } [ U \mid X _ { t } = x ] = ( t \Sigma - ( 1 - t ) \mathrm { I d } ) \left( ( 1 - t ) ^ { 2 } \mathrm { I d } + t ^ { 2 } \Sigma \right) ^ { - 1 } x .\tag{80}
$$

Diagonalizing this expression and diferentiating at fixed x gives equation 49 and equation 50; setting $\Sigma = \mathrm { I d }$ recovers equa<sup>ti</sup>on 51. □

Th<sub>e</sub> ind<sub>epe</sub>nd<sub>e</sub>nt fi<sub>e</sub>ld i<sub>s a</sub>l<sub>so e</sub>x<sub>ac</sub>tl<sub>y</sub> th<sub>e</sub> fi<sub>e</sub>ld ind<sub>uce</sub>d b<sub>y s</sub>in<sub>g</sub>l<sub>e</sub>t<sub>o</sub>n di<sub>sc</sub>r<sub>e</sub>t<sub>e</sub> minib<sub>a</sub>t<sub>c</sub>h OT<sub>,</sub> b<sub>ecause</sub> th<sub>e so</sub>l<sub>e</sub> source and tar<sub>g</sub>et observations must be <sub>p</sub>aired. This does not identif<sub>y</sub> it with Gaussian <sub>p</sub>lu<sub>g</sub>-in OT at $n = 1$ <sub>,</sub> f<sub>or</sub> whi<sub>c</sub>h <sub>a</sub>n <sub>u</sub>nbi<sub>ase</sub>d <sub>sa</sub>m<sub>p</sub>l<sub>e co</sub>v<sub>a</sub>ri<sub>a</sub>n<sub>ce</sub> i<sub>s u</sub>nd<sub>e</sub>fin<sub>e</sub>d<sub>.</sub>

Acceleration along flow trajectories. For a time-dependent velocity field $v _ { t }$ <sub>,</sub> d<sub>e</sub>fin<sub>e</sub> it<sub>s</sub> m<sub>a</sub>t<sub>e</sub>ri<sub>a</sub>l <sub>acce</sub>l<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n b<sub>y</sub>

$$
a _ { t } [ v ] ( x ) : = \partial _ { t } v _ { t } ( x ) + D _ { x } v _ { t } ( x ) v _ { t } ( x ) .\tag{81}
$$

This is the acceleration of a trajectory satisfying ${ \dot { x } } _ { t } = v _ { t } ( x _ { t } )$ . In <sub>p</sub>articular<sub>,</sub> the <sub>p</sub>o<sub>p</sub>ulation Gaussian OT field follows straight displacement trajectories and therefore satisfie $a _ { t } [ v ^ { \star } ] ( x ) = 0$

F<sub>or</sub> th<sub>e</sub> <sub>p</sub>l<sub>ug-</sub>i<sub>n</sub> fi<sub>e</sub>ld<sub>,</sub> l<sub>e</sub>t

$$
b _ { i } ( t ) : = \frac { \sigma _ { i } - 1 } { m _ { i } ( t ) }\tag{82}
$$

and write overdots for time derivatives. Ex<sub>p</sub>andin<sub>g</sub> e<sub>q</sub>uation 81 usin<sub>g</sub> e<sub>q</sub>uation <sub>4</sub>6 <sub>g</sub>ives

$$
\begin{array} { l } { { \displaystyle \left[ a _ { t } [ v _ { n } ] ( x ) \right] _ { i } = \frac { x _ { i } } { n } \bigg [ \dot { \kappa } _ { i } ( t ) + 2 b _ { i } ( t ) \kappa _ { i } ( t ) } } \\ { { \displaystyle \qquad + \sum _ { j = 1 } ^ { d } \frac { \dot { \theta } _ { i j } ( t ) - ( b _ { i } ( t ) + b _ { j } ( t ) ) \theta _ { i j } ( t ) } { m _ { i } ( t ) ^ { 3 } m _ { j } ( t ) ^ { 3 } } x _ { j } ^ { 2 } \bigg ] + o ( n ^ { - 1 } ) . } } \end{array}\tag{83}
$$

For inde<sub>p</sub>endent flow matchin<sub>g,</sub> the corres<sub>p</sub>ondin<sub>g</sub> ex<sub>p</sub>ression is exact:

$$
[ a _ { t } [ v ^ { \mathrm { i n d } } ] ( x ) ] _ { i } = \frac { \sigma _ { i } ^ { 2 } } { q _ { i } ( t ) ^ { 2 } } x _ { i } .\tag{84}
$$

Wh<sub>e</sub>n $\Sigma = \mathrm { I d }$ , write $q ( t ) = ( 1 - t ) ^ { 2 } + t ^ { 2 }$ <sub>.</sub> Th<sub>e</sub> th<sub>ree acce</sub>l<sub>era</sub>ti<sub>on</sub> fi<sub>e</sub>ld<sub>s s</sub>i<sub>mp</sub>lif<sub>y</sub> t<sub>o</sub>

$$
a _ { t } [ v ^ { \star } ] ( x ) = 0 , \qquad a _ { t } [ v _ { n } ] ( x ) = \frac { 2 \| x \| ^ { 2 } - ( d - 1 ) } { 2 n } x + o ( n ^ { - 1 } ) , \qquad a _ { t } [ v ^ { \mathrm { i n d } } ] ( x ) = \frac { x } { q ( t ) ^ { 2 } } .\tag{85}
$$

The <sub>p</sub>lu<sub>g</sub>-in material acceleration a<sub>g</sub>rees with the Eulerian derivative in e<sub>q</sub>uation <sub>4</sub>8 to order $1 / n ,$ b<sub>ecause</sub> th<sub>e</sub> <sub>co</sub>nv<sub>ec</sub>tiv<sub>e</sub> t<sub>e</sub>rm i<sub>s o</sub>f <sub>o</sub>rd<sub>e</sub>r $1 / n ^ { 2 }$ in th<sub>e s</sub>t<sub>a</sub>nd<sub>a</sub>rdiz<sub>e</sub>d <sub>case.</sub>

Fin<sub>a</sub>ll<sub>y,</sub> l<sub>e</sub>t $x _ { t }$ denote the flow trajectory initialized at $x _ { 0 } ,$ <sub>,</sub> <sub>a</sub>nd d<sub>e</sub>fin<sub>e</sub> it<sub>s</sub> int<sub>eg</sub>r<sub>a</sub>t<sub>e</sub>d <sub>squa</sub>r<sub>e</sub>d <sub>acce</sub>l<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n b<sub>y</sub>

$$
\mathcal { A } [ \boldsymbol { v } ; \boldsymbol { x } _ { 0 } ] : = \int _ { 0 } ^ { 1 } \| a _ { t } [ \boldsymbol { v } ] ( \boldsymbol { x } _ { t } ) \| ^ { 2 } \mathrm { d } t .\tag{86}
$$

F<sub>o</sub>r <sub>s</sub>t<sub>a</sub>nd<sub>a</sub>rdiz<sub>e</sub>d <sub>e</sub>nd<sub>po</sub>int<sub>s,</sub> $x _ { t } ^ { \star } = x _ { 0 } , x _ { t } ^ { \mathrm { i n d } } = \sqrt { q ( t ) }$ x<sub>0</sub>, and $x _ { t , n } ^ { \mathrm { p l u g } } = x _ { 0 } + O ( n ^ { - 1 } )$ . Consequent<sup>l</sup>y,

$$
\begin{array} { r } { A [ v ^ { \star } ; x _ { 0 } ] = 0 , } \end{array}\tag{87}
$$

$$
\mathcal { A } [ v _ { n } ; x _ { 0 } ] = \frac { \| x _ { 0 } \| ^ { 2 } \left( 2 \| x _ { 0 } \| ^ { 2 } - ( d - 1 ) \right) ^ { 2 } } { 4 n ^ { 2 } } + o ( n ^ { - 2 } ) ,\tag{88}
$$

$$
\mathcal { A } [ v ^ { \mathrm { i n d } } ; x _ { 0 } ] = \left( 2 + \frac { 3 \pi } { 4 } \right) \| x _ { 0 } \| ^ { 2 } .\tag{89}
$$

Thus the leading plug-in acceleration energy is nonincreasing in n and decays as $n ^ { - 2 }$ <sub>,</sub> while inde<sub>p</sub>endent FM incurs order-one acceleration and exact OT incurs none. For ever<sub>y</sub> fixed $x _ { 0 } \neq 0 ,$ <sub>,</sub> ind<sub>e e</sub>nd<sub>e</sub>nt FM th<sub>e</sub>r<sub>e</sub>f<sub>o</sub>r<sub>e</sub> has the largest of these three acceleration energies for all suficiently large n. This comparison concerns the controlled large-n expansion and does not assert exact monotonicity of the finite-n Gaussian plug-in field.

If th<sub>e</sub> initi<sub>a</sub>l <sub>co</sub>nditi<sub>o</sub>n i<sub>s</sub> it<sub>se</sub>lf r<sub>a</sub>nd<sub>o</sub>m<sub>,</sub> $X _ { 0 } \sim { \mathcal { N } } ( 0 , \operatorname { I d } )$ <sub>,</sub> th<sub>en</sub> $R : = \| X _ { 0 } \| ^ { 2 } \sim \chi _ { d } ^ { 2 }$ <sub>,</sub> <sub>w</sub>ith

$$
\mathbb { E } [ R ] = d , \qquad \mathbb { E } [ R ^ { 2 } ] = d ( d + 2 ) , \qquad \mathbb { E } [ R ^ { 3 } ] = d ( d + 2 ) ( d + 4 ) .\tag{90}
$$

Conse<sub>q</sub>uentl<sub>y,</sub> assumin<sub>g</sub> the <sub>p</sub>lu<sub>g</sub>-in ex<sub>p</sub>ansion above holds in $L ^ { 1 }$ with res<sub>p</sub>ect to $X _ { 0 }$

$$
\mathbb { E } [ \mathcal { A } [ v ^ { \star } ; X _ { 0 } ] ] = 0 ,\tag{91}
$$

$$
\mathbb { E } [ A [ v _ { n } ; X _ { 0 } ] ] = \frac { d \left( d ^ { 2 } + 1 8 d + 4 1 \right) } { 4 n ^ { 2 } } + o ( n ^ { - 2 } ) ,\tag{92}
$$

$$
\mathbb { E } \left[ \mathcal { A } [ v ^ { \mathrm { i n d } } ; X _ { 0 } ] \right] = \left( 2 + \frac { 3 \pi } { 4 } \right) d .\tag{93}
$$

Thus<sub>,</sub> at fixed dimension<sub>,</sub> the ex<sub>p</sub>ected <sub>p</sub>lu<sub>g</sub>-in acceleration ener<sub>gy</sub> a<sub>g</sub>ain deca<sub>y</sub>s as $n ^ { - 2 }$ <sub>, w</sub>h<sub>ereas</sub> th<sub>e</sub> independent-coupling energy remains order one in n.

## C ImageNet-256

## C.1 Implementation Details

Training details. In tables 6 and 7, we show training configurations across our SiT (Ma et al., 2024) and DMF (Lee et al., 2026) ex<sub>p</sub>eriments for B/2, L/2 and XL/2 model scales. All Ima<sub>g</sub>eNet-256 ex<sub>p</sub>eriments use mini-batch settin<sub>g</sub> of 256 batch size. DMF (Lee et al., 2026) conditions the encoder on time t and adds additional time r to <sub>co</sub>nditi<sub>o</sub>n it<sub>s</sub> d<sub>eco</sub>d<sub>e</sub>r<sub>,</sub> t<sub>u</sub>rnin<sub>g a</sub> fl<sub>o</sub>w m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> m<sub>o</sub>d<sub>e</sub>l int<sub>o a</sub> fl<sub>o</sub>w m<sub>ap.</sub> F<sub>o</sub>ll<sub>o</sub>win<sub>g</sub> th<sub>e</sub>ir <sub>se</sub>t-<sub>up,</sub> w<sub>e a</sub>l<sub>so se</sub>l<sub>ec</sub>t logit-normal distribution to sample (t, r) pairs using the time proposal parameters in table 6. DMF is trained b<sub>y</sub> finetunin<sub>g</sub> a <sub>p</sub>retrained SiT model via self-distillation meanflow objective (Gen<sub>g</sub> et al., 2025) with a dia<sub>g</sub>onal <sub>sp</sub>lit <sub>o</sub>f 0<sub>.5.</sub>

Table 6 ImageNet-256 implementation details across model scales.
<table><tr><td></td><td>B/2</td><td>L/2</td><td>XL/2</td></tr><tr><td colspan="4">Backbone</td></tr><tr><td>Resolution</td><td>256 × 256</td><td>256 × 256</td><td>256 × 256</td></tr><tr><td>Params (M)</td><td>130</td><td>458</td><td>675</td></tr><tr><td>FLOPS (G)</td><td>23.1</td><td>80.7</td><td>118.6</td></tr><tr><td>Hidden dim.</td><td>768</td><td>1024</td><td>1152</td></tr><tr><td>Heads</td><td>12</td><td>16</td><td>16</td></tr><tr><td>Patch size</td><td>2 × 2</td><td>2 × 2</td><td>2 × 2</td></tr><tr><td>Sequence length</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Layers</td><td>12</td><td>24</td><td>28</td></tr><tr><td colspan="4">Flow Matching (SiT) (Ma et al., 2024)</td></tr><tr><td>Training iterations</td><td>800K</td><td>800K</td><td>800K</td></tr><tr><td>Epochs</td><td>160</td><td>160</td><td>160</td></tr><tr><td>Class dropout probability</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td colspan="4">Flow Map (DMF) (Lee et al., 2026)</td></tr><tr><td>DMF depth</td><td>8</td><td>18</td><td>20</td></tr><tr><td>Training iterations</td><td>400K</td><td>400K</td><td>400K</td></tr><tr><td>Epochs</td><td>80</td><td>80</td><td>80</td></tr><tr><td>Class dropout probability</td><td></td><td>0.1</td><td></td></tr><tr><td>Time proposal µFM</td><td></td><td>0.0</td><td></td></tr><tr><td>Time proposal  $( \mu _ { \mathrm { M F } } ^ { ( 1 ) } , \mu _ { \mathrm { M F } } ^ { ( 2 ) } )$ </td><td></td><td>(0.4, −1.2)</td><td></td></tr><tr><td>Model guidance scale ω</td><td>0.5</td><td>0.6</td><td>0.6</td></tr><tr><td>Guidance interval</td><td>[0.0, 1.0]</td><td>[0.0, 0.7]</td><td>[0.0,0.7]</td></tr></table>

Table 7 Optimization hyperparameters, shared across all model scales and both training stages.
<table><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Learning rate</td><td>1e-4</td></tr><tr><td>Adam  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9,0.95)</td></tr><tr><td>Adam €</td><td>1e-8</td></tr><tr><td>Weight decay</td><td>0.0</td></tr><tr><td>EMA decay rate</td><td>0.9999</td></tr></table>

![](images/ddb2aa28f121c916ce592a49ccb5febf93b7c86e6e34ebcead29235891b2ea35.jpg)  
Figure 10 Sweeping $\beta _ { m }$ f<sub>o</sub>r <sub>c</sub>l<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l OT<sub>.</sub>

Compute budget. In table 8, we show cost per optimizer step for each of the coupling plans. We observe that class-conditional OT and GT add a modest cost per optimizer step, with GT being slightly higher than <sub>c</sub>l<sub>ass</sub>-<sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l d<sub>ue</sub> t<sub>o</sub> n<sub>o</sub>t b<sub>e</sub>in<sub>g</sub> r<sub>es</sub>tri<sub>c</sub>t<sub>e</sub>d <sub>pe</sub>r <sub>c</sub>l<sub>ass.</sub>

Table 8 Compute cost per optimizer step of SiT-B/2 training on ImageNet-256 for the three coupling plans. The second column is the increase over the inde<sub>p</sub>endent cou<sub>p</sub>lin<sub>g</sub>.
<table><tr><td>Coupling</td><td>ms / step</td><td>↑cost</td></tr><tr><td>Independent</td><td>155.5</td><td>一</td></tr><tr><td>Class-cond. OT</td><td>157.9</td><td> $\uparrow 1 . 6 \%$ </td></tr><tr><td>GT</td><td>160.4</td><td> $\uparrow 3 . 1 \%$ </td></tr></table>

Choice of $\beta$ for class-conditional OT. We choose $\beta$ such that it <sub>p</sub>revents an<sub>y</sub> cross-class cou<sub>p</sub>lin<sub>g,</sub> thereb<sub>y</sub> com<sub>p</sub>utin<sub>g</sub> trans<sub>p</sub>ort <sub>p</sub>lan within each class followin<sub>g</sub> (Chen<sub>g</sub> and Schwin<sub>g</sub>, 2025). In <sub>p</sub>ractice, we define $\beta =$ $\begin{array} { r } { \beta _ { m } \cdot \frac { 1 } { B ^ { 2 } } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { B } \lVert x _ { i } - x _ { j } \rVert _ { 2 } ^ { 2 } } \end{array}$ . We sweep $\beta _ { m }$ <sub>a</sub>nd <sub>s</sub>h<sub>o</sub>w <sub>a</sub>n<sub>a</sub>l<sub>ys</sub>i<sub>s</sub> in fi<sub>gu</sub>r<sub>e</sub> $^ { 1 0 , }$ choosin<sub>g</sub> $\beta _ { m } = 1 0$

Text-conditioning for ImageNet. Every ImageNet-1k image is paired with the captions taken from VisualLayer (2024) dataset. Each ca<sub>p</sub>tion is encoded once with the frozen DFN5B CLIP ViT-H/14 text tower into its <sub>p</sub>ooled, ℓ<sub>2</sub>-normalised 1024-d text embedding $c \in \mathbb { R } ^ { 1 0 2 4 }$

## C.2 Additional results on ImageNet-256

GT preserves mode stability. Figure 11 sweeps w for fixed noise and class. Under the independent coupling, increasin<sub>g g</sub>uidance results in abru<sub>p</sub>t chan<sub>g</sub>e in com<sub>p</sub>osition and <sub>p</sub>ose<sub>,</sub> e.<sub>g</sub>. from a <sub>p</sub>ortrait to a full-bod<sub>y</sub> view. Under GT the object remains stable for the full range of w.

Guidance interval tuning. We show that GT achieves strongest results overall with applied guidance interval tunin<sub>g</sub> across B/2<sub>,</sub> L/2 and XL/2 scales in table 10.

Table 9 FID ↓ and $\mathrm { F D } _ { \mathrm { D I N O v 2 } } \downarrow$ vs. guidance scale w for independent, class-conditional OT and GT couplings across SiT scales (64 Euler steps), under the full guidance interval [0, 1].
<table><tr><td>SiT-B/2</td><td colspan="13"></td></tr><tr><td>Coupling</td><td>Metric</td><td>w=1.0</td><td>1.5</td><td>2.0</td><td>2.5</td><td>3.0</td><td>3.5</td><td>4.0</td><td>5.0</td><td>6.0</td><td>8.0</td><td>10.0</td></tr><tr><td>Independent</td><td>FID FDDINOv2</td><td>26.44 605.11</td><td>6.68 337.16</td><td>5.16 212.66</td><td>7.93 159.87</td><td>11.04 139.10</td><td>13.65 132.46</td><td>15.76 132.27</td><td>18.63 139.64</td><td>20.42 149.79</td><td>22.40 169.51</td><td>23.28 189.02</td></tr><tr><td>Class-cond. OT</td><td>FID FDDINOv2</td><td>26.72 607.86</td><td>6.70 340.46</td><td>5.08 215.04</td><td>7.81 160.77</td><td>10.90 139.17</td><td>13.47 131.64</td><td>15.59 131.41</td><td>18.44 138.27</td><td>20.28 148.37</td><td>22.25 168.91</td><td>23.12 189.17</td></tr><tr><td>GT</td><td>FID FDDINOv2</td><td>31.16 636.79</td><td>8.13 365.26</td><td>4.15 226.75</td><td>5.88 161.94</td><td>8.72 132.62</td><td>11.32 120.25</td><td>13.41 116.36</td><td>16.51 119.84</td><td>18.62 128.33</td><td>20.81 148.82</td><td>21.62 171.99</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td></tr><tr><td>Coupling</td><td></td><td>Metric</td><td>w=1.0</td><td>1.5</td><td>1.75</td><td>2.0</td><td>2.5</td><td>3.0</td><td>3.5</td><td>4.0</td><td></td></tr><tr><td>Independent</td><td>FID FDDINOv2</td><td></td><td>13.85 360.44</td><td>2.95 154.77</td><td>3.86 115.56</td><td>5.85 95.16</td><td>10.10 83.28</td><td>13.53 86.46</td><td>16.03 94.58</td><td>17.89 103.47</td><td></td></tr><tr><td>Class-cond. OT</td><td>FID</td><td>FDDINOv2</td><td>13.96 360.38</td><td>2.98 154.83</td><td>3.84 115.16</td><td>5.81 95.01</td><td>10.02 83.41</td><td>13.42 86.38</td><td>15.93 94.01</td><td>17.79 102.64</td><td></td></tr><tr><td>GT</td><td>FID</td><td>FDDINOv2</td><td>17.72 392.20</td><td>2.96 170.86</td><td>2.69 123.70</td><td>3.88 97.58</td><td>7.38 76.33</td><td>10.66 73.48</td><td>13.25 77.32</td><td>15.17 83.28</td><td></td></tr><tr><td colspan="10">SiT-XL/2</td></tr><tr><td>Coupling</td><td>Metric</td><td>w=1.0</td><td>1.5</td><td>1.75</td><td>2.0</td><td>2.25</td><td>2.5</td><td>3.0</td><td>3.5</td><td>4.0</td><td></td></tr><tr><td>Independent</td><td>FID FDDINOv2</td><td>12.58 328.83</td><td>2.78</td><td>3.98</td><td>6.10</td><td>8.32</td><td>10.41 77.76</td><td>13.76 83.29</td><td>16.14 92.02</td><td>17.87 101.67</td><td></td></tr><tr><td>GT</td><td>FID</td><td>16.11</td><td>135.91 2.63</td><td>101.78 2.59</td><td>85.35 3.87</td><td>78.40 5.65</td><td>7.47</td><td>10.70</td><td>13.16</td><td>15.09</td><td></td></tr><tr><td></td><td>FDDINOv2</td><td>357.67</td><td>150.28</td><td>108.56</td><td>85.80</td><td>73.95</td><td>69.01</td><td>68.44</td><td>73.27</td><td>80.05</td><td></td></tr></table>

## D Single-cell Datasets

PBMC3K. 2,638 peripheral blood mononuclear cells from a healthy donor across 8 cell types.

Dentate gyrus. 18,213 cells from the developing mouse hippocampus (La Manno et al., 2018), annotated with 14 ce<sup>ll</sup> t<sub>yp</sub>es.

HLCA. 584,944 human lung cells from 486 individuals across 49 datasets (Sikkema et al., 2023), annotated with 50 ce<sup>ll</sup> t<sub>yp</sub>es.

Implementation Details. We add couplings on top of the uni-modal generation set-up described in Palma et al. (2025). Within the sin<sub>g</sub>le-cell dataset, CFGen s<sub>p</sub>lits data into 90% trainin<sub>g</sub> and 10% validation. Evaluation is <sub>p</sub>erformed on the corres<sub>p</sub>ondin<sub>g</sub> held-out test data. Since CFGen (Palma et al., 2025) considers inter<sub>p</sub>olant $\boldsymbol { x } _ { t } = t \boldsymbol { x } _ { 1 } + \sigma _ { t } \boldsymbol { \epsilon } , \boldsymbol { \epsilon } \sim \mathcal { N } ( 0 , I _ { d } )$ <sub>,</sub> we im<sub>p</sub>lement baseline with linear inter<sub>p</sub>olant used in our ima<sub>g</sub>e ex<sub>p</sub>eriments (CFGen-linear) which trains CFGen with inde<sub>p</sub>endent cou<sub>p</sub>lin<sub>g</sub>. We then com<sub>p</sub>are both baselines to GT trans<sub>p</sub>ort. Our evaluation scri t uses 0, 1, 2 seeds

CFGen s<sub>p</sub>lits its trainin<sub>g</sub> into two sta<sub>g</sub>es: 1) encodin<sub>g</sub> sin<sub>g</sub>le-cell data with an RNA autoencoder and 2) trainin<sub>g</sub> a <sub>co</sub>nditi<sub>o</sub>n<sub>a</sub>l fl<sub>o</sub>w m<sub>a</sub>t<sub>c</sub>hin m<sub>o</sub>d<sub>e</sub>l in l<sub>a</sub>t<sub>e</sub>nt <sub>s ace.</sub> W<sub>e s</sub>h<sub>o</sub>w h <sub>e</sub>r <sub>a</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> f<sub>o</sub>r tr<sub>a</sub>inin <sub>a</sub>nd b<sub>ac</sub>kb<sub>o</sub>n<sub>e</sub> model in tables 11 and 12. Note that PBMC3k and Dentate gyrus use resnet small and HLCA uses resnet big.

![](images/70c5f9b494bfcbfda9d13c0163a2956c9a9c8959fab83e2c06478c237fca96f5.jpg)  
Figure 11 SiT-XL/2 curated samples for class 339 sorrel across guidance scales w and coupling choices. Under GT the sample object characteristics remain stable, while for independent and class-conditional object changes its composition, pose, background at higher w.

Table 10 FID ↓ across model scales and couplings, under the full guidance interval and tuned interval [0, 0.7].
<table><tr><td></td><td></td><td></td><td colspan="5">Full</td><td colspan="5">GI [0, 0.7]</td></tr><tr><td>Model</td><td>Coupling</td><td> $w { = } 1 . 0$ </td><td>1.5</td><td>2.0</td><td>2.5</td><td>3.0</td><td>4.0</td><td>1.5</td><td>2.0</td><td>2.5</td><td>3.0</td><td>4.0</td></tr><tr><td></td><td>Independent</td><td>26.44</td><td>6.68</td><td>5.16</td><td>7.93</td><td>11.04</td><td>15.76</td><td>11.93</td><td>5.95</td><td>3.95</td><td>3.72</td><td>5.07</td></tr><tr><td>SiT-B/2</td><td>Class-cond. OT</td><td>26.72</td><td>6.70</td><td>5.08</td><td>7.81</td><td>10.90</td><td>15.59</td><td>12.13</td><td>6.00</td><td>3.92</td><td>3.68</td><td>5.00</td></tr><tr><td></td><td>GT</td><td>31.16</td><td>8.13</td><td>4.15</td><td>5.88</td><td>8.72</td><td>13.41</td><td>15.63</td><td>8.03</td><td>4.71</td><td>3.52</td><td>3.70</td></tr><tr><td>SiT-L/2</td><td>Independent</td><td>13.85</td><td>2.95</td><td>5.85</td><td>10.10</td><td>13.53</td><td>17.89</td><td>4.53</td><td>2.35</td><td>2.62</td><td>3.65</td><td>6.00</td></tr><tr><td></td><td>GT</td><td>17.72</td><td>2.96</td><td>3.88</td><td>7.38</td><td>10.66</td><td>15.17</td><td>6.64</td><td>2.96</td><td>2.17</td><td>2.48</td><td>4.02</td></tr><tr><td>SiT-XL/2</td><td>Independent</td><td>12.58</td><td>2.78</td><td>6.10</td><td>10.41</td><td>13.76</td><td>17.87</td><td>3.96</td><td>2.16</td><td>2.63</td><td>3.73</td><td>6.04</td></tr><tr><td></td><td>GT</td><td>16.11</td><td>2.63</td><td>3.87</td><td>7.47</td><td>10.70</td><td>15.09</td><td>5.89</td><td>2.61</td><td>2.01</td><td>2.40</td><td>3.91</td></tr></table>

## E Additional Background

## E.1 Classifier and classifier-free guidance

Classifier Guidance. By applying Bayes’ Rule we can write

$$
p _ { t } ( x \mid c ) = { \frac { p _ { t } ( c \mid x ) p _ { t } ( x ) } { p _ { t } ( c ) } }\tag{94}
$$

Takin<sub>g</sub> a lo<sub>g</sub>arithm of each side we <sub>g</sub>et

$$
\log p _ { t } ( x \mid c ) = \log p _ { t } ( c \mid x ) + \log p _ { t } ( x ) - \log p _ { t } ( c )\tag{95}
$$

<sup>B</sup>y app<sup>l</sup>y<sup>in</sup>g $\nabla _ { x }$ t<sub>o</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es</sub> w<sub>e</sub> <sub>ge</sub>t

$$
\nabla _ { \boldsymbol { x } } \log p _ { t } ( \boldsymbol { x } \mid \boldsymbol { c } ) = \nabla _ { \boldsymbol { x } } \log p _ { t } ( \boldsymbol { c } \mid \boldsymbol { x } ) + \nabla _ { \boldsymbol { x } } \log p _ { t } ( \boldsymbol { x } ) - \underbrace { \nabla _ { \boldsymbol { \alpha } } \log p _ { t } ( \boldsymbol { c } ) } _ { \mathrm { ~ } }\tag{96}
$$

![](images/ae54976eff27b2635b6f79c30bf631a6a64c8f75d550bf3819b1f4bb89860758.jpg)  
flow time t (1 = noise, 0 = data)

Figure 12 Measuring yo-yo efect and prediction gap across trajectory using SiT-XL/2 (64 steps).  
Table 11 Training configuration for uni-modal generation with CFGen.
<table><tr><td>Parameter</td><td>Autoencoder</td><td>Latent Flow Matching</td></tr><tr><td>Batch size</td><td>256</td><td>256 (64 for PBMC3K)</td></tr><tr><td>Epochs</td><td>300</td><td>1500</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td></tr><tr><td>Train/validation split</td><td>90% / 10%</td><td>90% / 10%</td></tr></table>

l<sub>ea</sub>din<sub>g</sub> t<sub>o</sub>

$$
\nabla _ { x } \log { p _ { t } ( x \mid c ) } = \nabla _ { x } \log { p _ { t } ( c \mid x ) } + \nabla _ { x } \log { p _ { t } ( x ) }\tag{97}
$$

U<sub>s</sub>i<sub>ng</sub> th<sub>e convers</sub>i<sub>on</sub> f<sub>ormu</sub>l<sub>a</sub> f<sub>or score we can</sub> f<sub>ur</sub>th<sub>er wr</sub>it<sub>e</sub> thi<sub>s as</sub>

$$
\boldsymbol { v } _ { c } ( \boldsymbol { x } , t , c ) = \boldsymbol { v } _ { u } ( \boldsymbol { x } _ { t } , t ) + a _ { t } \nabla _ { \boldsymbol { x } } \log p _ { t } ( c | \boldsymbol { x } )\tag{98}
$$

Cl<sub>ass</sub>ifi<sub>e</sub>r <sub>gu</sub>id<sub>a</sub>n<sub>ce co</sub>n<sub>s</sub>tr<sub>uc</sub>t<sub>s a</sub> v<sub>e</sub>l<sub>oc</sub>it<sub>y</sub> fi<sub>e</sub>ld b<sub>y e</sub>nh<sub>a</sub>n<sub>c</sub>in<sub>g c</sub>l<sub>ass</sub>ifi<sub>e</sub>r $p _ { t } ( c \vert x )$ with scaling factor w which we refer to as <sub>g</sub>uidance scale<sub>, y</sub>ieldin<sub>g</sub>

$$
\boxed { v _ { c } ^ { C F } ( x , t , c ) = v _ { u } ( x _ { t } , t ) + w a _ { t } \nabla _ { x } \log p _ { t } ( c \mid x ) }\tag{99}
$$

<sub>w</sub>h<sub>ere</sub> $\begin{array} { r } { a _ { t } = \frac { 1 - t } { t } } \end{array}$ <sub>a</sub>nd $\begin{array} { r } { b _ { t } = \frac { 1 } { t } } \end{array}$

The main issue with usin<sub>g</sub> classifier <sub>g</sub>uidance is the need to train $p _ { t } ( c \vert x )$ <sub>c</sub>l<sub>ass</sub>ifi<sub>er, w</sub>hi<sub>c</sub>h d<sub>ras</sub>ti<sub>ca</sub>ll<sub>y</sub> i<sub>ncreases</sub> trainin<sub>g</sub> com<sub>p</sub>ute. In the followin<sub>g</sub> section<sub>,</sub> we derive classifier-free <sub>g</sub>uidance.

Classifier-free Guidance. Classifier-free guidance takes a step further to construct a guided field that does not re<sub>q</sub>uire additional trainin<sub>g</sub> of classifier $p _ { t } ( c \mid x )$

Table 12 CFGen ResNet Backbone.
<table><tr><td>Hyperparameter</td><td>ResNet Small</td><td>ResNet Big</td></tr><tr><td>Hidden dimension</td><td>32</td><td>64</td></tr><tr><td>Residual blocks</td><td>3</td><td>3</td></tr><tr><td>Embedding dimension</td><td>20</td><td>100</td></tr><tr><td>Dropout probability</td><td>0.0</td><td>0.0</td></tr><tr><td>Condition dropout probability</td><td>0.2</td><td>0.2</td></tr></table>

W<sub>e ca</sub>n <sub>a</sub>dd $b _ { t } x _ { t }$ <sub>a</sub>nd <sub>su</sub>btr<sub>ac</sub>t fr<sub>o</sub>m th<sub>e</sub> RHS <sub>o</sub>f <sub>equa</sub>ti<sub>o</sub>n <sub>99</sub>

$$
\begin{array} { r l } & { v ^ { \mathrm { C F G } } ( x _ { t } , t , c ) = v _ { u } ( x _ { t } , t ) + w a _ { t } \nabla _ { x } \log p _ { t } ( c \mid x _ { t } ) } \\ & { \phantom { = } = v _ { u } ( x _ { t } , t ) + w a _ { t } \nabla _ { x } \log \frac { p _ { t } ( x _ { t } \mid c ) p _ { t } ( c ) } { p _ { t } ( x _ { t } ) } } \\ & { \phantom { = } = v _ { u } ( x _ { t } , t ) + w \big [ a _ { t } \nabla _ { x } \log p _ { t } ( x _ { t } \mid c ) + \underbrace { a _ { t } \nabla _ { \ast } \log p _ { t } ( c ) } _ { = v _ { u } ( x _ { t } , t ) + w \big [ \underbrace { a _ { t } \nabla _ { x } \log p _ { t } ( x _ { t } ) } _ { v _ { c } ( x _ { t } , t , c ) } \big ] } } \\ & { \phantom { = = } = v _ { u } ( x _ { t } , t ) + w \big [ \underbrace { a _ { t } \nabla _ { x } \log p _ { t } ( x _ { t } \mid c ) + b _ { t } x _ { t } } _ { v _ { c } ( x _ { t } , t , c ) } - \underbrace { \big ( a _ { t } \nabla _ { x } \log p _ { t } ( x _ { t } ) + b _ { t } x _ { t } \big ) } _ { v _ { u } ( x _ { t } , t ) } \big ] } \end{array}\tag{100}
$$

<sub>w</sub>hi<sub>c</sub>h l<sub>ea</sub>d<sub>s</sub> t<sub>o</sub>

$$
\boxed { v _ { c } ^ { C F G } ( x , t , c ) = v _ { u } ( x _ { t } , t ) + w \underbrace { \left( v _ { c } ( x , t , c ) - v _ { u } ( x , t ) \right) } _ { g ( x , t , c ) } }\tag{101}
$$

![](images/e3625fd67065a4d73347f9c6b81224c32b7e0408e9a120e35c22f23ec16bc058.jpg)  
Figure 13 Samples generated with SiT-L/2, Euler (16 steps), w = 4.0

## G Class-conditional OT Uncurated Samples

![](images/d1ef6ac1d991e3feabf08c73a55b3dd68e8df256a7c75eb2f001430e207889c4.jpg)  
Figure 14 Samples generated with SiT-L/2, Euler (16 steps), w = 4.0

![](images/0934cc823332c709442a7b6f06acefe1d7ed9de93f79b7706cd1be9551abba8c.jpg)  
Figure 15 Samples generated with SiT-L/2, Euler (16 steps), w = 4.0

## I Text-conditioned ImageNet-256 Uncurated Samples

We show samples for the following prompts read left to right:{two mittens with colorful yarn on a bed, a white keyboard with a small white keypad, a sink with a white marble top and a black base, a green and purple flower with a black center, three monkeys sitting on a wooden bench, a black and white photo of a stethoscope, a dog is looking at ducks in a pond, a woman feeding her baby with a bottle, a bowl of food, a dog sitting on the grass with a red building in the background, a rocky cliff, a snake is laying on top of hay in a cage, a plate of food on a table, a white monkey hanging on a wooden pole, casio dg-2000 boombox, a dog laying in the grass, a bathroom with a toilet and a bathtub with gummy bears, a chameleon is sitting on a branch with green leaves, a group of masks with different colors and designs, a monkey sitting in a tree with leaves, a person walking along a road near a body of water, a wolf laying on the ground, a black and white photo of a wheel, a large dam with a large waterfall in front of it, a lab coat with a picture of a man in a lab coat, a female swimmer in the pool with a yellow cap, a brown and white dog with a collar on, a plate with shrimp and bacon, a pair of scissors on a wooden table, a bird with a long beak sitting on a branch, a spider sits on its web in the sun, a tree with a branch, a group of people standing in a line, a pool table in an empty building with a green light, panasonic pd-wg-g1, a meerkat standing on a rock in a zoo, a small hamster sleeping in a person’s hand, a woman looking at a dinosaur in a museum, a knitting dish cloth and a knitting needle, a stone wall in the middle of a yard, a white and yellow sea slug on a coral reef, a large cicada sitting on a person’s hand, two women in kimono, a vintage sewing machine sitting in the grass, a couple standing in front of a yurt, a man lifting a barbell on a competition stage, a circular clock with a white circle in the middle, a military vehicle with a gun mounted on top, three graduates pose for a photo in blue graduation gowns, two cars driving on a race track, a woman holding a large fruit, a large black and white whale with its tail out in the water, a wall of banjos hanging on a wall, a bowl of guacamole with a tortilla chip on top, a close up of a dog with a collar, two dogs are standing in the dirt, a dock with a concrete wall and palm trees, a plate of mashed potatoes, a pair of sunglasses on the ground, a large building with many people walking around it, a small orange fish in a bowl, a close up of a metal fan with a metal cover, three monkeys sitting on a log, a skillet with food in it}

![](images/35819e0960ae3c2967ef21980494ca72156cd6991a93e7b9bfe61b5869283215.jpg)  
Figure 16 Samples generated with SiT-B/2, Euler (16 steps), w = 4.0

![](images/8a47a388e2b602afa7625cb38676f21e5e706d5162b0e7d32c8b36362db5e760.jpg)  
Figure 17 Samples generated with SiT-B/2, Euler (16 steps), w = 4.0