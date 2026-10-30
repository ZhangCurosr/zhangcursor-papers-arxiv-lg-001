# A Deep Learning Framework for Quantile Demand Forecasting in E-commerce with Doubly Censored Data

Zi<sub>we</sub>n Y<sub>e</sub><sup>∗</sup>   
guz<sup>i</sup>yzw@gma<sup>il</sup>.com   
Meit<sub>u</sub>an   
Beijing, C<sup>h</sup>ina   
Jiawen Wei<sup>†</sup>   
weijiawen@meituan.com   
Meit<sub>u</sub>an   
Beijing, C<sup>h</sup>ina   
Ch<sub>en</sub> Ch<sub>en</sub>   
<sub>c</sub>h<sub>enc</sub>h<sub>en</sub>11<sub>@me</sub>it<sub>uan.com</sub>   
Meit<sub>u</sub>an   
Beijing, C<sup>h</sup>ina   
<sup>T</sup>en<sub>g</sub> <sup>S</sup>on<sub>g</sub>   
<sup>son</sup>g<sup>ten</sup>g<sup>02@meituan.com</sup>   
Meit<sub>u</sub>an   
Beijing, C<sup>h</sup>ina   
G<sub>u</sub>an<sub>g</sub>r<sub>u</sub>i Ma   
ma<sub>g</sub>r@connect.ust.hk   
Meit<sub>u</sub>an   
Beijing, C<sup>h</sup>ina

## Abstract

In recent <sub>y</sub>ears<sub>,</sub> a <sub>g</sub>rowin<sub>g</sub> number of online retail com<sub>p</sub>anies have <sub>em</sub>b<sub>race</sub>d d<sub>eep</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>or</sub> d<sub>eman</sub>d f<sub>orecas</sub>ti<sub>ng,</sub> l<sub>everag-</sub> in<sub>g</sub> their scalabilit<sub>y</sub> in <sub>p</sub>rocessin<sub>g</sub> lar<sub>g</sub>e-scale datasets. Accurate d<sub>eman</sub>d f<sub>orecas</sub>t<sub>s</sub> <sub>ena</sub>bl<sub>e</sub> i<sub>mprove</sub>d <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>an</sub>d <sub>a</sub>ll<sub>ow</sub> <sub>opera</sub>ti<sub>ona</sub>l <sup>fl</sup>exi<sup>b</sup>i<sup>l</sup>ity to a<sup>d</sup>just sa<sup>l</sup>es accor<sup>d</sup>ing<sup>l</sup>y. Moreover, quanti<sup>l</sup>e estimation has <sub>g</sub>ained wides<sub>p</sub>read a<sub>pp</sub>lication as it ofers valuable uncertaint<sub>y</sub> <sub>q</sub>uantification<sub>,</sub> thereb<sub>y</sub> enhancin<sub>g</sub> the robustness of decision-<sub>ma</sub>ki<sub>ng processes.</sub> H<sub>owever,</sub> li<sub>m</sub>it<sub>e</sub>d <sub>researc</sub>h h<sub>as a</sub>dd<sub>resse</sub>d th<sub>e</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve quan</sub>tifi<sub>ca</sub>ti<sub>on o</sub>f <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y w</sub>h<sub>en</sub> t<sub>arge</sub>t l<sub>a</sub>b<sub>e</sub>l<sub>s are am-</sub> bi<sub>guous.</sub> I<sub>n re</sub>t<sub>a</sub>il<sub>, ac</sub>t<sub>ua</sub>l <sub>sa</sub>l<sub>es can</sub> b<sub>e cappe</sub>d b<sub>y s</sub>t<sub>oc</sub>k <sub>an</sub>d b<sub>oos</sub>t<sub>e</sub>d b<sub>y</sub> tem<sub>p</sub>orar<sub>y p</sub>romotions<sub>,</sub> whereas the tar<sub>g</sub>et is underl<sub>y</sub>in<sub>g</sub> de-<sub>man</sub>d<sub>.</sub> S<sub>uc</sub>h <sub>scenar</sub>i<sub>os are c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub>d b<sub>y</sub> d<sub>ou</sub>bl<sub>e censor</sub>i<sub>ng an</sub>d no dee<sub>p</sub>-learnin<sub>g p</sub>i<sub>p</sub>eline <sub>y</sub>et exists for such data. In this <sub>p</sub>a<sub>p</sub>er<sub>,</sub> we i<sub>n</sub>t<sub>ro</sub>d<sub>uce a s</sub>t<sub>ra</sub>i<sub>g</sub>htf<sub>orwar</sub>d f<sub>ramewor</sub>k f<sub>or</sub> l<sub>earn</sub>i<sub>ng quan</sub>til<sub>es</sub> f<sub>rom</sub> d<sub>ou</sub>bl<sub>y-censore</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons an</sub>d <sub>propose a</sub> t<sub>a</sub>il<sub>ore</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> metric. Our a<sub>pp</sub>roach is su<sub>pp</sub>orted b<sub>y</sub> theoretical <sub>g</sub>uarantees<sub>,</sub> and <sub>we emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t <sub>our me</sub>t<sub>r</sub>i<sub>c corre</sub>l<sub>a</sub>t<sub>es we</sub>ll <sub>w</sub>ith th<sub>e orac</sub>l<sub>e ca</sub>lib<sub>ra</sub>ti<sub>on measure.</sub> E<sub>xper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>resu</sub>lt<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>our</sub> estimator sur<sub>p</sub>asses existin<sub>g</sub> baselines. Finall<sub>y,</sub> inventor<sub>y</sub> simulati<sub>ons</sub> <sub>an</sub>d <sub>rea</sub>l<sub>-wor</sub>ld <sub>app</sub>li<sub>ca</sub>ti<sub>ons</sub> <sub>con</sub>fi<sub>rm</sub> th<sub>e</sub> <sub>prac</sub>ti<sub>ca</sub>l <sub>u</sub>tilit<sub>y</sub> <sub>o</sub>f our method in retail o<sub>p</sub>erations.

## CCS Concepts

• Computing methodologies → Neural networks; Supervised learning by regression.

## Keywords

Deep Learning, Inventory Management, Censored Demand, Quantil<sub>e</sub> R<sub>eg</sub>r<sub>ess</sub>i<sub>o</sub>n

Ziwen Ye, Jiawen Wei, Ten<sub>g</sub> Son<sub>g</sub>, Chen Chen, and Guan<sub>g</sub>rui Ma. 2026. A Deep Learning Framework for Quantile Demand Forecasting in E-commerce with Doubly Censored Data. In Proceedings ofthe 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 11 pages. https://doi. or<sub>g</sub>/10.1145/3799682.3840587

## 1 Introduction

In su<sub>pp</sub>l<sub>y</sub> chain mana<sub>g</sub>ement<sub>, q</sub>uantile re<sub>g</sub>ression is hi<sub>g</sub>hl<sub>y</sub> val-<sub>ue</sub>d f<sub>or</sub> it<sub>s</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on-</sub>f<sub>ree na</sub>t<sub>ure an</sub>d <sub>a</sub>bilit<sub>y</sub> t<sub>o es</sub>ti<sub>ma</sub>t<sub>e uncer-</sub> taint<sub>y</sub>. Retail <sub>g</sub>iants like JD and Amazon utilize dee<sub>p</sub> learnin<sub>g</sub>-based <sub>q</sub>uantile estimation for inventor<sub>y</sub> o<sub>p</sub>timization[25, 28, 29]. How-<sub>ever, w</sub>h<sub>en sa</sub>l<sub>es</sub> d<sub>a</sub>t<sub>a are censore</sub>d<sub>, suc</sub>h <sub>as</sub> d<sub>ue</sub> t<sub>o s</sub>t<sub>oc</sub>k<sub>ou</sub>t<sub>s or</sub> <sub>p</sub>romotions<sub>, q</sub>uantile re<sub>g</sub>ression estimates can become biased. As illustrated in Figure 1, right-censoring happens with perishable it<sub>ems, w</sub>h<sub>ere</sub> l<sub>ow s</sub>t<sub>oc</sub>k <sub>causes</sub> f<sub>requen</sub>t <sub>s</sub>t<sub>oc</sub>k<sub>ou</sub>t<sub>s an</sub>d <sub>concea</sub>l<sub>s</sub> t<sub>rue</sub> d<sub>eman</sub>d<sub>.</sub> O<sub>n</sub> th<sub>e</sub> fli<sub>p s</sub>id<sub>e,</sub> l<sub>e</sub>ft<sub>-censor</sub>i<sub>ng occurs w</sub>ith <sub>promo</sub>ti<sub>ona</sub>l <sub>pro</sub>d<sub>uc</sub>t<sub>s,</sub> <sub>w</sub>h<sub>ere</sub> t<sub>emporary</sub> <sub>promo</sub>ti<sub>ons</sub> l<sub>ea</sub>d t<sub>o</sub> <sub>o</sub>b<sub>serve</sub>d <sub>sa</sub>l<sub>es</sub> <sub>sur-</sub> <sub>p</sub>assin<sub>g</sub> actual demand. Dail<sub>y</sub> necessities t<sub>yp</sub>icall<sub>y</sub> have suficient <sub>s</sub>t<sub>oc</sub>k <sub>an</sub>d <sub>rare</sub>l<sub>y</sub> <sub>go</sub> <sub>on</sub> <sub>promo</sub>ti<sub>on,</sub> <sub>so</sub> th<sub>e</sub>i<sub>r</sub> <sub>sa</sub>l<sub>es</sub> d<sub>a</sub>t<sub>a</sub> <sub>usua</sub>ll<sub>y</sub> <sub>re</sub>fl<sub>ec</sub>t<sub>s</sub> th<sub>e</sub> <sub>ac</sub>t<sub>ua</sub>l d<sub>eman</sub>d<sub>.</sub> Wh<sub>en</sub> <sub>r</sub>i<sub>g</sub>ht <sub>an</sub>d l<sub>e</sub>ft <sub>censor</sub>i<sub>ng</sub> <sub>coex</sub>i<sub>s</sub>t i<sub>n</sub> <sub>one</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>,</sub> it’<sub>s</sub> <sub>ca</sub>ll<sub>e</sub>d d<sub>ou</sub>bl<sub>y</sub> <sub>censor</sub>i<sub>ng.</sub>

![](images/63877f6466e4fe0028bf820cb6c17dd8701caf3fb8920c650ca067d216182be4.jpg)

![](images/9e9496e56ddded8f0bb098ff96e461dea4c1e9299286fd7b2be5bc3eb6a5a179.jpg)

![](images/b9d2a3bc26209a7c5609d544514d7640a92e42ff190f5aac7b34a32f3178665d.jpg)  
Figure 1: Censored data example.

U<sub>s</sub>i<sub>ng or</sub> di<sub>scar</sub>di<sub>ng</sub> d<sub>ou</sub>bl<sub>y censore</sub>d d<sub>a</sub>t<sub>a</sub> b<sub>o</sub>th di<sub>s</sub>t<sub>or</sub>t f<sub>orecas</sub>t<sub>s:</sub> th<sub>e</sub> f<sub>ormer</sub> fit<sub>s</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>on</sub> bi<sub>ase</sub>d l<sub>a</sub>b<sub>e</sub>l<sub>s,</sub> th<sub>e</sub> l<sub>a</sub>tt<sub>er</sub> di<sub>scar</sub>d<sub>s usa</sub>bl<sub>e</sub> information[40]. In <sub>p</sub>ractice, retailers often revise sales usin<sub>g</sub> ad h<sub>oc</sub> b<sub>us</sub>i<sub>ness</sub> <sub>ru</sub>l<sub>es,</sub> <sub>ye</sub>t <sub>w</sub>ith<sub>ou</sub>t t<sub>rue</sub> d<sub>eman</sub>d th<sub>ey</sub> <sub>canno</sub>t <sub>ver</sub>if<sub>y</sub> th<sub>e</sub> <sub>accuracy or</sub> f<sub>ee</sub>d th<sub>e rev</sub>i<sub>se</sub>d d<sub>a</sub>t<sub>a</sub> b<sub>ac</sub>k i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> f<sub>orecas</sub>ti<sub>ng p</sub>i<sub>pe</sub>li<sub>ne.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> d<sub>eman</sub>d f<sub>orecas</sub>ti<sub>ng</sub> <sub>s</sub>till l<sub>ac</sub>k<sub>s</sub> <sub>an</sub> <sub>en</sub>d<sub>-</sub>t<sub>o-en</sub>d t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>me</sub>th<sub>o</sub>d f<sub>or</sub> d<sub>ou</sub>bl<sub>y censore</sub>d d<sub>a</sub>t<sub>a an</sub>d <sub>an appropr</sub>i<sub>a</sub>t<sub>e me</sub>t<sub>r</sub>i<sub>c</sub> t<sub>o</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>un</sub>d<sub>er</sub> thi<sub>s</sub> <sub>scenar</sub>i<sub>o.</sub> S<sub>urv</sub>i<sub>va</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f<sub>ers</sub> <sub>a</sub> di<sub>rec</sub>t <sub>para</sub>ll<sub>e</sub>l<sub>:</sub> <sub>on</sub>l<sub>y</sub> <sub>a</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> lif<sub>e</sub>ti<sub>me</sub> i<sub>s</sub> <sub>o</sub>b<sub>serve</sub>d<sub>,</sub> <sub>pro</sub>d<sub>uc</sub>i<sub>ng</sub> <sub>r</sub>i<sub>g</sub>ht<sub>-</sub> censored res<sub>p</sub>onses. Re-wei<sub>g</sub>htin<sub>g</sub> techni<sub>q</sub>ues develo<sub>p</sub>ed for this <sup>∗</sup>Thi<sub>s wor</sub>k <sub>was</sub> d<sub>one w</sub>hil<sub>e</sub> th<sub>e au</sub>th<sub>or was a</sub>t M<sub>e</sub>it<sub>uan.</sub> <sup>†</sup><sub>co</sub>rr<sub>espo</sub>ndin<sub>g</sub> A<sub>u</sub>th<sub>o</sub>r<sub>.</sub>

<sub>se</sub>tti<sub>ng</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>e</sub> <sub>our</sub> <sub>un</sub>ifi<sub>e</sub>d f<sub>ramewor</sub>k<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h t<sub>ra</sub>i<sub>ns</sub> <sub>any</sub> d<sub>eep</sub> network end-to-end with the Doubl<sub>y</sub> Portno<sub>y</sub> Estimator (DPE). Li<sup>k</sup>ewise, w<sup>h</sup>i<sup>l</sup>e quanti<sup>l</sup>e <sup>f</sup>orecasts are usua<sup>ll</sup>y ju<sup>d</sup>ge<sup>d</sup> <sup>b</sup>y ca<sup>l</sup>i<sup>b</sup>ration <sub>an</sub>d <sub>s</sub>h<sub>arpness, no ca</sub>lib<sub>ra</sub>ti<sub>on</sub> t<sub>es</sub>t <sub>ex</sub>i<sub>s</sub>t<sub>s</sub> f<sub>or</sub> d<sub>ou</sub>bl<sub>y censore</sub>d d<sub>a</sub>t<sub>a.</sub> Borrowin<sub>g</sub> a<sub>g</sub>ain from survival methodolo<sub>gy,</sub> we <sub>p</sub>ro<sub>p</sub>ose Extended D<sub>-</sub>C<sub>a</sub>lib<sub>ra</sub>ti<sub>on</sub> t<sub>o</sub> <sub>assess</sub> th<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>o</sub>f <sub>compe</sub>ti<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>un</sub>d<sub>er</sub> d<sub>ou</sub>bl<sub>e</sub> <sub>censor</sub>i<sub>ng.</sub>

I<sub>n</sub> <sub>conc</sub>l<sub>us</sub>i<sub>on,</sub> thi<sub>s</sub> <sub>paper</sub> <sub>presen</sub>t<sub>s</sub> <sub>a</sub> <sub>un</sub>ifi<sub>e</sub>d f<sub>ramewor</sub>k f<sub>or</sub> t<sub>ra</sub>i<sub>n</sub> i<sub>ng</sub> <sub>an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> d<sub>eep</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>un</sub>d<sub>er</sub> d<sub>ou</sub>bl<sub>y</sub> <sub>censore</sub>d d<sub>a</sub>t<sub>a.</sub> Al<sub>-</sub> th<sub>oug</sub>h <sub>mo</sub>ti<sub>va</sub>t<sub>e</sub>d b<sub>y</sub> <sub>e-commerce</sub> d<sub>eman</sub>d f<sub>orecas</sub>ti<sub>ng,</sub> th<sub>e</sub> f<sub>rame-</sub> work is domain-a<sub>g</sub>nostic and can be extended to an<sub>y</sub> settin<sub>g</sub> with similar censorin<sub>g</sub> <sub>p</sub>atterns. Our contributions are as follows.

• We introduce an end-to-end al<sub>g</sub>orithm that trains dee<sub>p</sub> mod <sub>e</sub>l<sub>s</sub> <sub>w</sub>ith th<sub>e</sub> DPE<sub>,</sub> <sub>an</sub>d <sub>prov</sub>id<sub>e</sub> th<sub>e</sub> fi<sub>rs</sub>t f<sub>orma</sub>l <sub>proo</sub>f <sub>o</sub>f it<sub>s</sub> <sub>properness</sub> <sub>un</sub>d<sub>er</sub> d<sub>ou</sub>bl<sub>y</sub> <sub>censore</sub>d d<sub>a</sub>t<sub>a,</sub> <sub>c</sub>l<sub>os</sub>i<sub>ng</sub> <sub>a</sub> l<sub>ong-s</sub>t<sub>an</sub>di<sub>ng</sub> theoretical <sub>g</sub>a<sub>p</sub> in <sub>q</sub>uantile estimation with incom<sub>p</sub>lete observations.

• To evaluate models trained under such censorin<sub>g</sub>, we <sub>p</sub>ro<sub>p</sub>ose E<sub>x</sub>t<sub>en</sub>d<sub>e</sub>d D<sub>-</sub>C<sub>a</sub>lib<sub>ra</sub>ti<sub>on, a new me</sub>t<sub>r</sub>i<sub>c</sub> f<sub>or</sub> d<sub>ou</sub>bl<sub>y censore</sub>d data<sub>,</sub> and <sub>p</sub>rove that it <sub>p</sub>recisel<sub>y</sub> <sub>g</sub>au<sub>g</sub>es the <sub>g</sub>a<sub>p</sub> between <sub>mo</sub>d<sub>e</sub>ll<sub>e</sub>d <sub>an</sub>d t<sub>rue</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>un</sub>d<sub>er</sub> <sub>m</sub>ild <sub>assump</sub>ti<sub>ons.</sub>

• We first validate the <sub>p</sub>ro<sub>p</sub>osed metric and al<sub>g</sub>orithm throu<sub>g</sub>h <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons,</sub> th<sub>en con</sub>fi<sub>rm</sub> th<sub>e</sub>i<sub>r e</sub>f<sub>ec</sub>ti<sub>veness on</sub> t<sub>wo</sub> l<sub>arge-</sub> <sub>sca</sub>l<sub>e rea</sub>l<sub>-wor</sub>ld d<sub>a</sub>t<sub>ase</sub>t<sub>s.</sub> I<sub>nven</sub>t<sub>ory s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons</sub> f<sub>ur</sub>th<sub>er</sub> d<sub>emon-</sub> strate their su<sub>p</sub>eriorit<sub>y</sub> in stock control<sub>,</sub> and online $\mathrm { A } / \mathrm { B }$ tests verif<sub>y</sub> their <sub>p</sub>ractical im<sub>p</sub>act.

## 2 Related Work

## 2.1 Probabilistic Demand Forecasting

Probabilistic demand forecastin<sub>g p</sub>la<sub>y</sub>s a crucial role in inventor<sub>y</sub> mana<sub>g</sub>ement across vario<sub>u</sub>s sectors. A widel<sub>y</sub> st<sub>u</sub>died <sub>p</sub>roblem in thi<sub>s</sub> d<sub>oma</sub>i<sub>n</sub> i<sub>s</sub> th<sub>e</sub> <sub>newsven</sub>d<sub>or</sub> <sub>pro</sub>bl<sub>em,</sub> <sub>w</sub>hi<sub>c</sub>h id<sub>en</sub>tifi<sub>es</sub> <sub>a</sub> <sub>spe-</sub> <sub>c</sub>ifi<sub>c</sub> <sub>quan</sub>til<sub>e</sub> <sub>o</sub>f th<sub>e</sub> d<sub>eman</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>as</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l i<sub>nven</sub>t<sub>ory</sub> level[2, 4]. Traditional a<sub>pp</sub>roaches for estimatin<sub>g</sub> demand <sub>q</sub>uan tiles t<sub>yp</sub>icall<sub>y</sub> rel<sub>y</sub> on deterministic forecasts and the historical residuals of <sub>p</sub>redicted values[21]. With the rise of dee<sub>p</sub> learnin<sub>g</sub>, <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>-</sub>b<sub>ase</sub>d d<sub>eman</sub>d f<sub>orecas</sub>ti<sub>ng</sub> h<sub>as</sub> b<sub>ecome</sub> i<sub>ncreas</sub>i<sub>ng</sub>l<sub>y</sub> <sub>p</sub>revalent. For exam<sub>p</sub>le<sub>,</sub> to ca<sub>p</sub>ture uncertaint<sub>y,</sub> Dee<sub>p</sub>AR em<sub>p</sub>lo<sub>y</sub>s a Recurrent Neural Network (RNN) to <sub>g</sub>enerate hidden states and <sub>es</sub>ti<sub>ma</sub>t<sub>es</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on parame</sub>t<sub>ers</sub> th<sub>roug</sub>h <sub>max</sub>i<sub>mum</sub> lik<sub>e</sub>lih<sub>oo</sub>d estimation[30]. This a<sub>pp</sub>roach re<sub>q</sub>uires assum<sub>p</sub>tions about the underl<sub>y</sub>in<sub>g</sub> data distribution.

In contrast<sub>,</sub> <sub>q</sub>uantile re<sub>g</sub>ression ofers a non-<sub>p</sub>arametric a<sub>pp</sub>roach for <sub>p</sub>robabilistic forecastin<sub>g</sub> that does not re<sub>q</sub>uire distributional assum<sub>p</sub>tions[20]. Buildin<sub>g</sub> on this foundation, Wen, Lim, and other researchers[22, 36] have desi<sub>g</sub>ned Se<sub>q</sub>2Se<sub>q</sub> models that directl<sub>y</sub> estimate <sub>q</sub>uantiles usin<sub>g q</sub>uantile loss as the loss function. How-<sub>ever,</sub> th<sub>ese mo</sub>d<sub>e</sub>l<sub>s</sub> t<sub>yp</sub>i<sub>ca</sub>ll<sub>y prov</sub>id<sub>e pre</sub>di<sub>c</sub>ti<sub>ons</sub> f<sub>or a</sub> fi<sub>xe</sub>d <sub>se</sub>t <sub>o</sub>f <sub>q</sub>uantiles. To enable arbitrar<sub>y</sub> <sub>q</sub>uantile forecastin<sub>g,</sub> a s<sub>p</sub>line-based neural network have been <sub>p</sub>ro<sub>p</sub>osed[5, 11, 25, 32]. These networks l<sub>everage</sub> th<sub>e</sub> fl<sub>ex</sub>ibilit<sub>y o</sub>f <sub>sp</sub>li<sub>nes</sub> t<sub>o</sub> i<sub>n</sub>t<sub>erpo</sub>l<sub>a</sub>t<sub>e</sub> b<sub>e</sub>t<sub>ween quan</sub>til<sub>es,</sub> th<sub>ere</sub>b<sub>y</sub> <sub>a</sub>ll<sub>ow</sub>i<sub>ng</sub> f<sub>or</sub> <sub>more</sub> <sub>granu</sub>l<sub>ar</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> f<sub>orecas</sub>t<sub>s.</sub>

I<sub>n</sub> th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> f<sub>orecas</sub>ti<sub>ng,</sub> th<sub>e</sub> th<sub>eory</sub> <sub>o</sub>f <sub>scor</sub>i<sub>ng</sub> <sub>ru</sub>l<sub>es</sub> <sub>serves</sub> <sub>as</sub> <sub>a</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l f<sub>ramewor</sub>k<sub>.</sub> A l<sub>oss</sub> f<sub>unc</sub>ti<sub>on</sub> that <sub>q</sub>ualifies as a <sub>p</sub>ro<sub>p</sub>er scorin<sub>g</sub> rule ensures that model trainin<sub>g</sub> with this function can elicit the true underl<sub>y</sub>in<sub>g</sub> distribution[12].

T<sub>o eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>veness o</sub>f dif<sub>eren</sub>t <sub>proper scor</sub>i<sub>ng ru</sub>l<sub>es, re-</sub> <sub>cen</sub>t <sub>s</sub>t<sub>u</sub>di<sub>es</sub> h<sub>ave w</sub>id<sub>e</sub>l<sub>y a</sub>d<sub>op</sub>t<sub>e</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on an</sub>d <sub>s</sub>h<sub>arpness as</sub> k<sub>ey</sub> evaluation criteria[8, 10, 32]. Our stud<sub>y</sub> also em<sub>p</sub>lo<sub>y</sub>s these metrics t<sub>o eva</sub>l<sub>ua</sub>t<sub>e mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>ormance.</sub>

## 2.2 Censored Quantile Regression

A<sub>s</sub> hi<sub>g</sub>hli<sub>g</sub>ht<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on, censore</sub>d d<sub>a</sub>t<sub>a rema</sub>i<sub>n</sub> l<sub>arge</sub>l<sub>y</sub> <sub>over</sub>l<sub>oo</sub>k<sub>e</sub>d i<sub>n</sub> d<sub>eman</sub>d f<sub>orecas</sub>ti<sub>ng.</sub> T<sub>o exp</sub>l<sub>o</sub>it <sub>suc</sub>h d<sub>a</sub>t<sub>a, we</sub> d<sub>raw</sub> <sub>upon</sub> <sub>surv</sub>i<sub>va</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s,</sub> <sub>a</sub> fi<sub>e</sub>ld <sub>w</sub>ith <sub>an</sub> <sub>ex</sub>t<sub>ens</sub>i<sub>ve</sub> lit<sub>era</sub>t<sub>ure</sub> <sub>on</sub> <sub>cen-</sub> sorin<sub>g</sub> (see [27, 35] for a com<sub>p</sub>rehensive surve<sub>y</sub>).

D<sub>eep</sub> l<sub>earn</sub>i<sub>ng</sub> h<sub>as</sub> <sub>recen</sub>tl<sub>y</sub> b<sub>een</sub> <sub>com</sub>bi<sub>ne</sub>d <sub>w</sub>ith <sub>surv</sub>i<sub>va</sub>l <sub>ana</sub>l<sub>y</sub> sis under ri<sub>g</sub>ht censorin<sub>g</sub> [19, 40]. Some studies use censored data durin<sub>g</sub> trainin<sub>g</sub> [1, 13]. In the statistical literature, doubl<sub>y</sub> censored data have been studied via <sub>q</sub>uantile re<sub>g</sub>ression [18, 23, 27], but few works link censored quantile regression to deep models. The qrnn package trains networks with left censoring [6]. Pearce et al. <sub>e</sub>xt<sub>e</sub>nd P<sub>o</sub>rtn<sub>oy</sub>’<sub>s</sub> E<sub>s</sub>tim<sub>a</sub>t<sub>o</sub>r t<sub>o</sub> tr<sub>a</sub>in d<sub>eep</sub> m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> <sub>v</sub>i<sub>a</sub> <sub>a</sub>n it<sub>e</sub>r<sub>a</sub>ti<sub>ve</sub> re-wei<sub>g</sub>htin<sub>g</sub> al<sub>g</sub>orithm [26]. Yana<sub>g</sub>isawa shows that Portno<sub>y</sub>’s Estimator is <sub>p</sub>ro<sub>p</sub>er<sub>,</sub> so the true distribution is recovered b<sub>y</sub> o<sub>p</sub>timization [39].

On the evaluation side, the C-index [17] <sub>q</sub>uantifies median discrimination onl<sub>y</sub>, whereas D-Calibration [16] assesses calibration <sub>un</sub>d<sub>er r</sub>i<sub>g</sub>ht <sub>censor</sub>i<sub>ng.</sub> N<sub>e</sub>ith<sub>er ex</sub>t<sub>en</sub>d<sub>s</sub> t<sub>o</sub> d<sub>ou</sub>bl<sub>y censore</sub>d d<sub>a</sub>t<sub>a.</sub> We close this <sub>g</sub>a<sub>p</sub> b<sub>y</sub> introducin<sub>g</sub> Extended D-Calibration<sub>,</sub> the first <sub>me</sub>t<sub>r</sub>i<sub>c</sub> t<sub>o eva</sub>l<sub>ua</sub>t<sub>e ca</sub>lib<sub>ra</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e presence o</sub>f b<sub>o</sub>th l<sub>e</sub>ft <sub>an</sub>d <sub>r</sub>i<sub>g</sub>ht censor<sup>i</sup>n<sub>g</sub>.

## 3 Preliminaries

## 3.1 Notation

We consider the task of usin<sub>g</sub> se<sub>q</sub>uential features � ∈ X to learn the <sub>q</sub>uantile function of demand <sub>�</sub> ∈ Y with a censorin<sub>g</sub> indicator Δ. Given a dataset $\mathcal { D } = \{ ( x _ { 1 } , y _ { 1 } ^ { * } , \Delta _ { 1 } ) , . . . , ( x _ { N } , y _ { N } ^ { * } , \Delta _ { N } ) \} , y ^ { * }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e o</sub>b<sub>serve</sub>d d<sub>eman</sub>d <sub>an</sub>d $y$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> t<sub>rue</sub> d<sub>eman</sub>d<sub>.</sub> Wh<sub>en no</sub> censor<sup>i</sup>n<sub>g</sub> occurs, $y ^ { * } = y$ <sub>an</sub>d <sub>we se</sub>t $\Delta = 0 .$ Let � and � be the ri<sub>g</sub>ht-<sub>censore</sub>d d<sub>eman</sub>d <sub>an</sub>d l<sub>e</sub>ft<sub>-censore</sub>d d<sub>eman</sub>d<sub>, respec</sub>ti<sub>ve</sub>l<sub>y.</sub> I<sub>n</sub> th<sub>e</sub> ri<sub>g</sub>ht-censorin<sub>g</sub> scenarios<sub>,</sub> we can onl<sub>y</sub> observe �<sub>,</sub> where we denote Δ as 1. Similarl<sub>y</sub>, Δ is denoted as 2 for left-censorin<sub>g</sub> scenarios, <sub>w</sub>h<sub>ere</sub> � i<sub>s</sub> <sub>o</sub>b<sub>serve</sub>d<sub>.</sub> Wh<sub>en</sub> <sub>no</sub> <sub>censor</sub>i<sub>ng</sub> <sub>occurs,</sub> th<sub>e</sub> <sub>�-quan</sub>til<sub>e</sub> <sub>o</sub>f <sub>�</sub> conditional on <sub>�</sub> is obtained b<sub>y</sub>:

$$
y _ { \tau } = F ^ { - 1 } ( \tau | x ) = \underset { z \in R } { \mathrm { a r g m i n } } \ \underset { Y \sim F } { \mathbb { E } } \left[ \rho _ { \tau } ( y , z ) \right]
$$

<sub>w</sub>h<sub>ere</sub> $\rho _ { \tau } ( y , z ) = ( \tau - \mathbb { 1 } \left[ y < z \right] ) ( y - z )$ i<sub>s</sub> th<sub>e</sub> <sub>quan</sub>til<sub>e</sub> l<sub>oss,</sub> $\tau \in ( 0 , 1 )$ is the <sub>q</sub>uantile level and 1 is the indicator function[20]. Quantile loss has been shown to be a <sub>p</sub>ro<sub>p</sub>er scorin<sub>g</sub> rule[12]. The formal d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>proper</sub> <sub>scor</sub>i<sub>ng</sub> <sub>ru</sub>l<sub>e</sub> <sub>un</sub>d<sub>er</sub> <sub>censore</sub>d d<sub>a</sub>t<sub>a</sub> i<sub>s</sub> <sub>g</sub>i<sub>ven</sub> as follows[39],

Definition 3.1. A scoring rule $S ( \hat { F } , ( y , \Delta ) )$ is proper if

$$
\mathbb { E } _ { y \sim Y } [ S ( \hat { F } , ( y , \Delta ) ) ] \ge \mathbb { E } _ { y \sim Y } [ S ( F , ( y , \Delta ) ) ] .
$$

Where � is the cumulative distribution function (CDF) and $\hat { F }$ i<sub>s</sub> th<sub>e es</sub>tim<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f �<sub>.</sub> In <sub>p</sub>r<sub>o</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> f<sub>o</sub>r<sub>ecas</sub>tin<sub>g,</sub> tr<sub>a</sub>inin<sub>g w</sub>ith <sub>a</sub> <sub>proper</sub> <sub>scor</sub>i<sub>ng</sub> <sub>ru</sub>l<sub>e</sub> i<sub>s</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d<sub>:</sub> if th<sub>e</sub> l<sub>oss</sub> i<sub>s</sub> <sub>no</sub>t <sub>proper,</sub> th<sub>e</sub> fitt<sub>e</sub>d di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $\hat { F }$ <sub>may</sub> d<sub>ev</sub>i<sub>a</sub>t<sub>e</sub> <sub>ar</sub>bit<sub>rar</sub>il<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> t<sub>rue</sub> $F .$

U<sub>n</sub>l<sub>ess s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>o</sub>th<sub>erw</sub>i<sub>se, we a</sub>d<sub>op</sub>t th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>surv</sub>i<sub>va</sub>l<sub>-ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>assump</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>censor</sub>i<sub>ng</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f th<sub>e</sub> t<sub>rue</sub> d<sub>eman</sub>d <sub>con</sub>di<sub>-</sub> ti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> f<sub>ea</sub>t<sub>ures.</sub> Thi<sub>s assump</sub>ti<sub>on can</sub> b<sub>e expresse</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

Assumption 3.1. [Independent censoring] $( L , R ) \perp \perp y | x$

Whil<sub>e uno</sub>b<sub>serve</sub>d <sub>con</sub>f<sub>oun</sub>d<sub>ers may v</sub>i<sub>o</sub>l<sub>a</sub>t<sub>e</sub> thi<sub>s</sub> i<sub>n prac</sub>ti<sub>ce, our</sub> covariates (e.<sub>g</sub>., <sub>p</sub>rice, <sub>p</sub>romotions) absorb the main sources of en-<sup>d</sup>o<sub>g</sub>ene<sup>i</sup>t<sub>y</sub>.

## 3.2 Evaluation

C<sub>a</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>an</sub>d <sub>s</sub>h<sub>arpness</sub> <sub>are</sub> th<sub>e</sub> t<sub>wo</sub> <sub>canon</sub>i<sub>ca</sub>l <sub>cr</sub>it<sub>er</sub>i<sub>a</sub> f<sub>or</sub> <sub>eva</sub>l<sub>u-</sub> atin<sub>g</sub> <sub>p</sub>robabilistic forecasts. For calibration<sub>,</sub> we first examine the <sub>uncensore</sub>d <sub>case.</sub> T<sub>o</sub> <sub>c</sub>h<sub>ec</sub>k <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>on</sub> <sub>an</sub> i<sub>n</sub>t<sub>erva</sub>l $I = [ \tau _ { k } , \tau _ { k + 1 } ]$ <sub>,</sub> we sim<sub>p</sub>l<sub>y</sub> count how man<sub>y</sub> observations f<sub>a</sub>ll i<sub>n</sub>t<sub>o</sub> th<sub>a</sub>t i<sub>n</sub>t<sub>erva</sub>l<sub>:</sub> $N _ { k } = \sum _ { i = 1 } ^ { N } \mathbb { 1 } ( \tau _ { i y } \in I )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\tau _ { i y } = \hat { F } ( y _ { i } | x _ { i } )$ i<sub>s</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f th<sub>e</sub> �<sub>-</sub>th <sub>o</sub>b<sub>serva</sub>ti<sub>on.</sub> If $N _ { k } / N$ e<sub>q</sub>ua<sup>l</sup>s $\tau _ { k + 1 } - \tau _ { k }$ <sub>,</sub> th<sub>e mo</sub>d<sub>e</sub>l i<sub>s ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>on</sub> �<sub>.</sub> Th<sub>e overa</sub>ll <sub>ca</sub>lib<sub>ra</sub>ti<sub>on error</sub> <sub>o</sub>f th<sub>e</sub> CDF i<sub>s</sub> th<sub>en</sub> <sub>measure</sub>d b<sub>y</sub>

$$
\mathrm { C a l i b r a t i o n } = \sum _ { k = 1 } ^ { K - 1 } ( \tau _ { k + 1 } - \tau _ { k } - \frac { N _ { k } } { N } ) ^ { 2 } .
$$

U<sub>n</sub>d<sub>er</sub> <sub>r</sub>i<sub>g</sub>ht<sub>-censor</sub>i<sub>ng,</sub> <sub>we</sub> <sub>canno</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>a</sub> <sub>censore</sub>d observation trul<sub>y</sub> falls inside �. Haider et al.[16] therefore re-wei<sub>g</sub>hts each ri<sub>g</sub>ht-censored observation b<sub>y</sub> its distance to the interval<sub>,</sub> usin<sub>g</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y mass:</sub>

$$
\begin{array} { l } { { \displaystyle N _ { k } ^ { R } = \sum _ { i = 1 } ^ { N } \underbrace { \{ \mathbb { 1 } ( \Delta _ { i } = 0 ) \mathbb { 1 } ( \tau _ { i y } \in I )  } } _ { \displaystyle { u n c e n s o r e d o b s e r v a t i o n } } }  \\ { { \displaystyle + \underbrace { \mathbb { 1 } ( \Delta _ { i } = 1 ) [ \frac { \tau _ { k + 1 } - \tau _ { i R } } { 1 - \tau _ { i R } } \mathbb { 1 } ( \tau _ { i R } \in I ) + \frac { \tau _ { k + 1 } - \tau _ { k } } { 1 - \tau _ { i R } } \mathbb { 1 } ( \tau _ { i R } < \tau _ { k } ) ] } _ { \displaystyle { r i g h t - c e n s o r e d o b s e r v a t i o n } } \} } } \end{array}
$$

<sub>a</sub>nd D-C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n i<sub>s</sub> th<sub>e</sub>n

$$
\mathrm { D - C a l i b r a t i o n } = \sum _ { k = 1 } ^ { K - 1 } ( \tau _ { k + 1 } - \tau _ { k } - \frac { N _ { k } ^ { R } } { N } ) ^ { 2 } .
$$

Yet ca<sup>l</sup>i<sup>b</sup>ration a<sup>l</sup>one <sup>d</sup>oes not su<sup>fi</sup>ce to ju<sup>d</sup>ge <sup>f</sup>orecast qua<sup>l</sup>ity [12]. When two models exhibit e<sub>q</sub>ual calibration, the one with the more concentrated <sub>p</sub>redictive distribution is <sub>p</sub>referred. Shar<sub>p</sub>ness<sub>,</sub> <sub>a proper</sub>t<sub>y o</sub>f th<sub>e</sub> f<sub>orecas</sub>t<sub>s a</sub>l<sub>one,</sub> i<sub>s measure</sub>d b<sub>y</sub> th<sub>e w</sub>idth <sub>o</sub>f th<sub>e</sub> central <sub>p</sub>redictive interval. For <sub>q</sub>uantiles $\tau _ { 1 }$ <sub>an</sub>d $\tau _ { 2 }$ <sub>w</sub>ith $\tau _ { 1 } ~ < ~ \tau _ { 2 }$ <sub>s</sub>h<sub>arpness</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub>

$$
\delta = \hat { F } ^ { - 1 } ( \tau _ { 2 } ) - \hat { F } ^ { - 1 } ( \tau _ { 1 } ) .
$$

<sub>w</sub>h<sub>ere</sub> $\hat { F }$ d<sub>e</sub>n<sub>o</sub>t<sub>es</sub> th<sub>e es</sub>tim<sub>a</sub>t<sub>e</sub>d CDF<sub>.</sub>

## 4 Methodology

## 4.1 Quantile Modeling

In <sub>s</sub>im<sub>u</sub>l<sub>a</sub>ti<sub>o</sub>n<sub>s we use a</sub>n MLP <sub>e</sub>n<sub>co</sub>d<sub>e</sub>r t<sub>o</sub> m<sub>ap</sub> f<sub>ea</sub>t<sub>u</sub>r<sub>es</sub> int<sub>o e</sub>mbedding representations E. For real-world data, demand patterns <sub>are</sub> <sub>more</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ca</sub>t<sub>e.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>a</sub>d<sub>op</sub>t <sub>a</sub> T<sub>rans</sub>f<sub>ormer</sub> <sub>enco</sub>d<sub>er</sub> that <sub>p</sub>rocesses historical sales<sub>,</sub> <sub>p</sub>rices and auxiliar<sub>y</sub> variables. It is worth notin<sub>g</sub> that alternative encoders such as Tem<sub>p</sub>oral Fusion Transformer[22], PatchTST[24] and TSMixer[7] can re<sub>p</sub>lace th<sub>e</sub> T<sub>rans</sub>f<sub>ormer as nee</sub>d<sub>e</sub>d<sub>.</sub> T<sub>o ena</sub>bl<sub>e ar</sub>bit<sub>rary-quan</sub>til<sub>e es</sub>ti<sub>ma-</sub> ti<sub>o</sub>n<sub>, we</sub> m<sub>o</sub>d<sub>e</sub>l th<sub>e</sub> in<sub>ve</sub>r<sub>se</sub> CDF <sub>as a p</sub>i<sub>ecew</sub>i<sub>se</sub> lin<sub>ea</sub>r f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n<sub>.</sub> Th<sub>e</sub> unit interval [0, 1) is s<sub>p</sub>lit into � e<sub>q</sub>ual se<sub>g</sub>ments b<sub>y</sub> $K + 1$ k<sub>no</sub>t<sub>s</sub> $\{ ( \tau _ { k } , \boldsymbol { \hat { y } } _ { k } ) \} _ { k = 1 } ^ { K + 1 } . \mathrm { A }$ neural network takes E as input and outputs a $K \cdot$ di<sub>mens</sub>i<sub>ona</sub>l <sub>vec</sub>t<sub>or</sub> �<sub>.</sub> Aft<sub>er a so</sub>ft<sub>p</sub>l<sub>us ac</sub>ti<sub>va</sub>ti<sub>on,</sub> � i<sub>s</sub> t<sub>rans</sub>f<sub>orme</sub>d i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>quan</sub>til<sub>e</sub> <sub>va</sub>l<sub>ues</sub> $\left\{ \hat { y } _ { k } \right\} _ { k = 1 } ^ { K }$ . For an<sub>y</sub> value <sub>�,</sub> we identif<sub>y</sub> the i<sub>n</sub>t<sub>erva</sub>l $I _ { q } = [ \hat { y } _ { k } , \hat { y } _ { k + 1 } ]$ such that <sub>�</sub> ∈ $I _ { q }$ and com<sub>p</sub>ute its <sub>q</sub>uantile as

$$
\begin{array} { l } { { \hat { \tau } _ { y } = \bigl ( 1 - w _ { y } \bigr ) \tau _ { k } + w _ { y } \tau _ { k + 1 } , } } \\ { { \nonumber } } \\ { { w _ { y } = \displaystyle \frac { y - \hat { y } _ { k } } { \hat { y } _ { k + 1 } - \hat { y } _ { k } } . } } \end{array}
$$

We name this a<sub>pp</sub>roach $q _ { - }$ search and use it throughout the remaind<sub>er</sub> <sub>o</sub>f th<sub>e</sub> <sub>paper.</sub>

## 4.2 Training

We em<sub>p</sub>lo<sub>y</sub> DPE to estimate the <sub>q</sub>uantile function under doubl<sub>y</sub> <sub>censore</sub>d d<sub>a</sub>t<sub>a.</sub> F<sub>or a g</sub>i<sub>ven quan</sub>til<sub>e</sub> l<sub>eve</sub>l $\tau \in ( 0 , 1 )$ <sub>,</sub> th<sub>e</sub> DPE l<sub>oss</sub> f<sub>or</sub> <sub>p</sub>redicted <sub>q</sub>uantile $y _ { \tau }$ i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as:</sub>

$$
\begin{array} { r l } & { S _ { D P E } ( y ^ { * } , y _ { \tau } , w _ { l } , w _ { r } , \tau , \Delta ) = \underbrace { \mathbb { 1 } ( \Delta = 0 ) \rho _ { \tau } ( y _ { \tau } , y ) } _ { L o s \rho f o r t h e u n c e n s o r e d s e n a r i } } \\ & { + \underbrace { \mathbb { 1 } ( \Delta = 1 ) \big [ w _ { r } \rho _ { \tau } ( y _ { \tau } , R ) + ( 1 - w _ { r } ) \rho _ { \tau } ( y _ { \tau } , z _ { \infty } ) \big ] } _ { L o s \rho c t h e r i g h t c e n s o r i n g s c e n a r i o } } \\ & { + \underbrace { \mathbb { 1 } ( \Delta = 2 ) \big [ w _ { l } \rho _ { \tau } ( y _ { \tau } , L ) + ( 1 - w _ { l } ) \rho _ { \tau } ( y _ { \tau } , z _ { - \infty } ) \big ] } _ { L o s s \rho o r t h e l e f t c e n s o r i n g s c e n a r i o } } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $z _ { \infty }$ <sub>an</sub>d ${ z - \infty }$ are suficientl<sub>y</sub> lar<sub>g</sub>e <sub>p</sub>ositive and ne<sub>g</sub>ative constants<sub>,</sub> res<sub>p</sub>ectivel<sub>y,</sub> and $w _ { l } , w _ { r } \in \left[ 0 , 1 \right]$ are redistrib<sub>u</sub>tion wei<sub>g</sub>hts.

To set the <sub>w</sub>ei<sub>g</sub>hts<sub>, w</sub>e examine the indicator f<sub>u</sub>nction $\mathbb { 1 } \left( y \right. <$ $y _ { \tau } )$ in the <sub>q</sub>uantile loss under censorin<sub>g</sub>. For left-censorin<sub>g,</sub> the <sub>o</sub>b<sub>serve</sub>d <sub>va</sub>l<sub>ue</sub> � <sub>serves as an upper</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> t<sub>rue</sub> d<sub>eman</sub>d $( y \le L )$ <sub>.</sub> Th<sub>us,</sub> if $y _ { \tau } > L ,$ <sub>, we</sub> h<sub>ave</sub> $y \le L < y _ { \tau }$ <sub>w</sub>ith certaint<sub>y,</sub> <sup>i</sup>m<sub>p</sub><sup>l</sup><sub>y</sub><sup>i</sup>n<sub>g</sub> $\mathbb { 1 } ( y < y _ { \tau } ) = 1$ and settin<sub>g</sub> $w _ { l } = 1$ . Con<sub>v</sub>ersel<sub>y,</sub> $\mathrm { i f } \ y _ { \tau } < L ,$ the indicator is ambi<sub>g</sub>uous because the true <sub>�</sub> ma<sub>y</sub> lie on either side <sub>o</sub>f $y _ { \tau }$ . Followin<sub>g</sub> Efron’s [34] redistribution <sub>p</sub>rinci<sub>p</sub>le, we im<sub>p</sub>ute the conditional ex<sub>p</sub>ectation:

$$
w _ { l } = \mathbb { E } [ \mathbb { 1 } \left( y > y _ { \tau } \right) \mid y < L ] = \frac { P ( y _ { \tau } < y < L ) } { P ( y < L ) } = \frac { \tau _ { L } - \tau } { \tau _ { L } } ,
$$

<sub>w</sub>h<sub>ere</sub> $\tau = F ( y _ { \tau } )$ <sub>an</sub>d $\tau _ { L } = F ( L )$ <sub>are</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve pro</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>u</sub>nder the tr<sub>u</sub>e demand CDF � . The com<sub>p</sub>lementar<sub>y</sub> wei<sub>g</sub>ht $( 1 - w _ { l } )$ i<sub>s re</sub>di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d t<sub>o</sub> ${ z - \infty } .$ <sub>, represen</sub>ti<sub>ng mass</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-censor</sub>i<sub>ng</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub> Th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht<sub>-censor</sub>i<sub>ng</sub> <sub>we</sub>i<sub>g</sub>ht $w _ { r }$ i<sub>s</sub> d<sub>er</sub>i<sub>ve</sub>d <sub>ana</sub>l<sub>ogous</sub>l<sub>y:</sub>

$$
w _ { r } = \operatorname { \mathbb { E } } [ \mathbb { 1 } \left( y < y _ { \tau } \right) \mid y > R ] = { \frac { P ( R < y < y _ { \tau } ) } { P ( y > R ) } } = { \frac { \tau - \tau _ { R } } { 1 - \tau _ { R } } } .
$$

Under the inde<sub>p</sub>endent censorin<sub>g</sub> assum<sub>p</sub>tion<sub>,</sub> and with <sub>��, ��</sub> defined with res<sub>p</sub>ect to the true CDF � <sub>,</sub> DPE induces a <sub>p</sub>ro<sub>p</sub>er scorin<sub>g</sub> rule for <sub>�</sub>-<sub>q</sub>uantile estimation. This result<sub>,</sub> formalized in Th<sub>eorem</sub> 1<sub>,</sub> i<sub>s</sub> t<sub>o</sub> <sub>our</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>properness</sub> <sub>guaran</sub>t<sub>ee</sub> f<sub>or</sub> <sub>a quan</sub>til<sub>e</sub> l<sub>oss accommo</sub>d<sub>a</sub>ti<sub>ng</sub> b<sub>o</sub>th l<sub>e</sub>ft <sub>an</sub>d <sub>r</sub>i<sub>g</sub>ht <sub>censor</sub>i<sub>ng.</sub>

Theorem 1. Under the independent censoring assumption, if �<sub>�</sub> and �<sub>�</sub> are correct, Doubly Portnoy’s Estimator is a proper scoring rule.

Proof<sub>.</sub> W<sub>e</sub> <sub>g</sub>i<sub>ve</sub> <sub>a</sub> <sub>p</sub>r<sub>oo</sub>f in A<sub>ppe</sub>ndix A<sub>.</sub>

Based on the wei<sub>g</sub>htin<sub>g</sub> strate<sub>gy</sub> described above<sub>,</sub> Theorem 1 establishes that DPE is a <sub>p</sub>ro<sub>p</sub>er scorin<sub>g</sub> rule for the tar<sub>g</sub>et <sub>q</sub>uantil<sub>es</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>re</sub>di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>we</sub>i<sub>g</sub>ht<sub>s</sub> <sub>�� an</sub>d $w _ { r }$ <sub>are</sub> <sub>spec</sub>ifi<sub>e</sub>d <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> th<sub>e</sub> t<sub>rue</sub> d<sub>eman</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> I<sub>n prac</sub>ti<sub>ce,</sub> h<sub>owever,</sub> th<sub>e</sub> cumulative <sub>p</sub>robabilities $\tau _ { L } = F ( L )$ <sub>an</sub>d $\tau _ { R } = F ( R )$ at the censorin<sub>g</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s are un</sub>k<sub>nown an</sub>d <sub>mus</sub>t b<sub>e es</sub>ti<sub>ma</sub>t<sub>e</sub>d f<sub>rom</sub> d<sub>a</sub>t<sub>a.</sub>

Algorithm 1: Iterative Re-weighting for DPE Training   
Data: Batch data $B = \{ ( x _ { j } , y _ { j } ^ { * } , \Delta _ { j } ) \} _ { j = 1 } ^ { b }$   
Model: Deep quantile model $M _ { \theta }$ <sub>w</sub>ith � k<sub>no</sub>t<sub>s</sub> $\{ \tau _ { i } \} _ { i = 1 } ^ { K }$   
Require: Target quantile $\tau ,$ <sup>l</sup>earn<sup>i</sup>n<sub>g</sub> rate �, conver<sub>g</sub>ence   
t<sub>o</sub>l<sub>erance �</sub>   
while max<sub>�</sub> $| w _ { r , j } ^ { ( t ) } - w _ { r , j } ^ { ( t - 1 ) } | > \epsilon$ or max<sub>�</sub> $| w _ { l , j } ^ { ( t ) } - w _ { l , j } ^ { ( t - 1 ) } | > \epsilon$   
do   
$/ / \mathsf { \Pi } \mathsf { E } { \mathsf { - s t e p } } \colon$ estimate redistribution weights   
(stop-gradient)   
for $j = 1 , \dotsc , b$ do   
$\{ \hat { y } _ { j , \tau _ { i } } \} _ { i = 1 } ^ { K } \gets M _ { \theta ^ { ( t - 1 ) } } ( x _ { j } ) ;$   
$\{ \tilde { y } _ { j , \tau _ { i } } \} _ { i = 1 } ^ { K } \gets \mathrm { S G } ( \{ \hat { y } _ { j , \tau _ { i } } \} _ { i = 1 } ^ { K } ) ;$   
if $\Delta _ { j } = 0$ then   
$\begin{array} { r } { \check { w } _ { r , j } ^ { ( t ) } \gets 1 ; w _ { l , j } ^ { ( t ) } \gets 1 ; } \end{array}$   
else if $\Delta _ { j } = 1$ then   
$\hat { \tau } _ { R } \gets q \_ s e a r c h ( \{ \tilde { y } _ { j , \tau _ { i } } \} _ { i = 1 } ^ { K } , R _ { j } ) ;$   
if $\hat { \tau } _ { R } \leq \tau$ then   
$w _ { r , j } ^ { ( t ) } \gets ( \tau - \hat { \tau } _ { R } ) / ( 1 - \hat { \tau } _ { R } ) ;$   
else   
$w _ { r , j } ^ { ( t ) } \gets 1 ;$   
end   
$w _ { l , j } ^ { ( t ) } \gets 1 ;$   
else if $\Delta _ { j } = 2$ then   
$\hat { \tau } _ { L } \gets \check { q } \_ s e a r c h ( \{ \tilde { y } _ { j , \tau _ { i } } \} _ { i = 1 } ^ { K } , L _ { j } ) ;$   
$\mathbf { i f } \ \tau \leq \hat { \tau } _ { L }$ then   
$w _ { l , j } ^ { ( t ) } \gets ( \hat { \tau } _ { L } - \tau ) / \hat { \tau } _ { L } ;$   
else   
$w _ { l , j } ^ { ( t ) } \gets 1 ;$   
end   
$w _ { r , j } ^ { ( t ) } \gets 1 ;$   
end   
// M-step: partial maximization via gradient   
descent   
$\begin{array} { r } { L ( \theta ) \gets \sum _ { j = 1 } ^ { b } S _ { D P E } ( y _ { j } ^ { * } , \hat { y } _ { j , \tau } , w _ { l , j } ^ { ( t ) } , w _ { r , j } ^ { ( t ) } , \tau , \Delta _ { j } ) ; } \end{array}$   
${ \boldsymbol { \theta } } ^ { ( t ) } \gets$ O<sub>p</sub>timizerU<sub>p</sub>date $( \theta ^ { ( t - 1 ) } , \nabla _ { \theta } L , \alpha )$   
end

T<sub>o</sub> im<sub>p</sub>l<sub>e</sub>m<sub>e</sub>nt DPE in <sub>p</sub>r<sub>ac</sub>ti<sub>ce,</sub> <sub>we</sub> <sub>e</sub>xt<sub>e</sub>nd th<sub>e</sub> it<sub>e</sub>r<sub>a</sub>ti<sub>ve</sub> r<sub>e</sub>-<sub>we</sub>i<sub>g</sub>htin<sub>g</sub> (IR) <sub>p</sub>rocedure of Pearce et al. [26] from the ri<sub>g</sub>ht-censored survival <sub>se</sub>tti<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> d<sub>ou</sub>bl<sub>y</sub> <sub>censore</sub>d d<sub>eman</sub>d f<sub>orecas</sub>ti<sub>ng</sub> <sub>scenar</sub>i<sub>o.</sub> W<sub>e</sub> <sub>ex</sub>t<sub>en</sub>d thi<sub>s</sub> <sub>proce</sub>d<sub>ure</sub> t<sub>o</sub> d<sub>ou</sub>bl<sub>y</sub> <sub>censore</sub>d d<sub>eman</sub>d f<sub>orecas</sub>ti<sub>ng</sub> <sub>an</sub>d <sub>a</sub>d<sub>ap</sub>t it f<sub>or</sub> d<sub>eep</sub> <sub>neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>s,</sub> <sub>as</sub> <sub>summar</sub>i<sub>ze</sub>d i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 1<sub>.</sub>

Theoretical justification. We now show that Algorithm 1 admits a <sub>p</sub>rinci<sub>p</sub>led <sub>p</sub>robabilistic inter<sub>p</sub>retation<sub>,</sub> extendin<sub>g</sub> the framework of Pearce et al. [26] to the doubl<sub>y</sub> censored settin<sub>g</sub>. Followin<sub>g</sub>

P<sub>earce e</sub>t <sub>a</sub>l<sub>., we assume</sub> th<sub>a</sub>t th<sub>e</sub> lik<sub>e</sub>lih<sub>oo</sub>d f<sub>or eac</sub>h <sub>uncensore</sub>d data<sub>p</sub>oint at <sub>q</sub>uantile level <sub>�</sub> follows an as<sub>y</sub>mmetric La<sub>p</sub>lace distribution (ALD) with location $\hat { y } _ { j , \ i }$ <sub>an</sub>d <sub>sca</sub>l<sub>e</sub> $\lambda = \sqrt { \tau ( 1 - \tau ) }$ <sub>.</sub> U<sub>n</sub>d<sub>er</sub> thi<sub>s</sub> <sub>assump</sub>ti<sub>on,</sub> th<sub>e</sub> l<sub>og-</sub>d<sub>ens</sub>it<sub>y correspon</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e nega</sub>ti<sub>ve c</sub>h<sub>ec</sub>k l<sub>oss</sub> u<sub>p</sub> to an additive constant [26, E<sub>q</sub>. 16]:

$$
\log f ( y _ { j } ; \hat { y } _ { j , \tau } , \lambda ) = - \rho _ { \tau } ( y _ { j } , \hat { y } _ { j , \tau } ) + \mathrm { c o n s t . }
$$

F<sub>or</sub> <sub>censore</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d lik<sub>e</sub>lih<sub>oo</sub>d b<sub>y</sub> redistributin<sub>g</sub> <sub>p</sub>robabilit<sub>y</sub> mass accordin<sub>g</sub> to the current wei<sub>g</sub>ht estimates. For a rig<sup>h</sup>t-censore<sup>d</sup> samp<sup>l</sup>e � wit<sup>h</sup> censoring time $R _ { j }$ th<sub>e co</sub>ntrib<sub>u</sub>ti<sub>o</sub>n i<sub>s</sub>:

$$
\begin{array} { r } { f _ { \mathrm { w e i g h t e d } } ( y _ { j } ; \hat { y } _ { j , \tau } , w _ { r , j } ) = f ( R _ { j } ; \hat { y } _ { j , \tau } ) ^ { w _ { r , j } } \cdot f ( z _ { \infty } ; \hat { y } _ { j , \tau } ) ^ { 1 - w _ { r , j } } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $z _ { \infty }$ is a suficientl<sub>y</sub> lar<sub>g</sub>e constant re<sub>p</sub>resentin<sub>g</sub> mass be<sub>y</sub>ond th<sub>e censor</sub>i<sub>ng</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub> Th<sub>e</sub> l<sub>e</sub>ft<sub>-censore</sub>d <sub>case</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>ana</sub>l<sub>o-</sub> gous<sup>l</sup>y. Com<sup>b</sup>ining a<sup>ll</sup> samp<sup>l</sup>es, t<sup>h</sup>e joint weig<sup>h</sup>te<sup>d</sup> <sup>l</sup>i<sup>k</sup>e<sup>l</sup>i<sup>h</sup>oo<sup>d</sup> is:

$$
\begin{array} { l } { \displaystyle p ( { \mathbf y } | { \mathbf X } , \boldsymbol { \theta } , { \mathbf w } _ { l } , { \mathbf w } _ { r } , \tau ) = \prod _ { i : \Delta _ { i } = 0 } f ( y _ { i } ; \hat { y } _ { i , \tau } ) } \\ { \displaystyle \qquad \times \prod _ { j : \Delta _ { j } = 1 } f _ { \mathrm { w e i g h t e d } } ( y _ { j } ; \hat { y } _ { j , \tau } , w _ { r , j } ) } \\ { \displaystyle \qquad \times \prod _ { k : \Delta _ { k } = 2 } f _ { \mathrm { w e i g h t e d } } ( y _ { k } ; \hat { y } _ { k , \tau } , w _ { l , k } ) . } \end{array}
$$

Takin<sub>g</sub> the ne<sub>g</sub>ative lo<sub>g</sub>-likelihood and substitutin<sub>g</sub> the ALD densit<sub>y</sub> recovers exactl<sub>y</sub> the DPE loss $S _ { D P E }$ (u<sub>p</sub> to an additive constant). Thus, minimizin<sub>g</sub> the DPE loss is e<sub>q</sub>uivalent to maximizin<sub>g</sub> th<sub>e we</sub>i<sub>g</sub>ht<sub>e</sub>d ALD lik<sub>e</sub>lih<sub>oo</sub>d<sub>.</sub> Thi<sub>s revea</sub>l<sub>s</sub> th<sub>a</sub>t Al<sub>gor</sub>ith<sub>m</sub> 1 <sub>can</sub> be interpreted as a generalized EM algorithm [26]: the E-step com<sub>p</sub>utes the redistribution wei<sub>g</sub>hts b<sub>y</sub> evaluatin<sub>g</sub> the conditional <sub>expec</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>curren</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>an</sub>d th<sub>e</sub> M<sub>-s</sub>t<sub>ep</sub> <sub>per</sub>f<sub>orms</sub> <sub>a</sub> <sub>par</sub>ti<sub>a</sub>l <sub>max</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>comp</sub>l<sub>e</sub>t<sub>e-</sub>d<sub>a</sub>t<sub>a</sub> l<sub>og-</sub>lik<sub>e</sub>lih<sub>oo</sub>d <sub>v</sub>i<sub>a gra</sub>di<sub>en</sub>t d<sub>escen</sub>t<sub>.</sub> Thi<sub>s pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on e</sub>l<sub>eva</sub>t<sub>es</sub> th<sub>e</sub> iterative re-wei<sub>g</sub>htin<sub>g</sub> strate<sub>gy</sub> above heuristic reasonin<sub>g,</sub> ensuring t<sup>h</sup>at eac<sup>h</sup> iteration yie<sup>ld</sup>s a we<sup>ll</sup>-<sup>d</sup>e<sup>fi</sup>ne<sup>d</sup> statistica<sup>l</sup> o<sup>b</sup>jective. Alth<sub>oug</sub>h <sub>we</sub>i<sub>g</sub>ht<sub>s are approx</sub>i<sub>ma</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e curren</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> th<sub>e</sub> <sub>g</sub>eneralized EM <sub>p</sub>rocedure conver<sub>g</sub>es to a stationar<sub>y</sub> <sub>p</sub>oint under mild conditions [38].

## 4.3 Evaluation

![](images/40e61c4e89799b14ad31e3021d6ad8ffbf7487e328146654c96baffd5922e128.jpg)  
Figure 2: The illustration of re-weighting strategy.

Having esta<sup>bl</sup>is<sup>h</sup>e<sup>d</sup> a training o<sup>b</sup>jective <sup>f</sup>or <sup>d</sup>ou<sup>bl</sup>y censore<sup>d d</sup>ata, <sub>we now</sub> t<sub>urn</sub> t<sub>o eva</sub>l<sub>ua</sub>ti<sub>on.</sub> St<sub>an</sub>d<sub>ar</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on me</sub>t<sub>r</sub>i<sub>cs assume</sub> f<sub>u</sub>ll<sub>y</sub> <sub>o</sub>b<sub>serve</sub>d <sub>ou</sub>t<sub>comes</sub> <sub>an</sub>d th<sub>us</sub> f<sub>a</sub>il <sub>un</sub>d<sub>er</sub> <sub>censor</sub>i<sub>ng.</sub> W<sub>e</sub> th<sub>ere</sub> f<sub>ore</sub> d<sub>eve</sub>l<sub>op</sub> E<sub>x</sub>t<sub>en</sub>d<sub>e</sub>d D<sub>-</sub>C<sub>a</sub>lib<sub>ra</sub>ti<sub>on, w</sub>hi<sub>c</sub>h <sub>genera</sub>li<sub>zes</sub> th<sub>e re-</sub> <sub>w</sub>ei<sub>g</sub>htin<sub>g</sub> <sub>p</sub>rinci<sub>p</sub>le of DPE to assessment.

F<sub>or</sub> <sub>uncensore</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> <sub>mem</sub>b<sub>ers</sub>hi<sub>p</sub> <sub>o</sub>f $\tau _ { y }$ in $I = \left[ \tau _ { k } , \tau _ { k + 1 } \right]$ i<sub>s cer</sub>t<sub>a</sub>i<sub>n, so ca</sub>lib<sub>ra</sub>ti<sub>on</sub> i<sub>s compu</sub>t<sub>e</sub>d di<sub>rec</sub>tl<sub>y.</sub> Wh<sub>en</sub> d<sub>a</sub>t<sub>a are</sub> d<sub>ou</sub>bl<sub>y</sub> censored<sub>,</sub> ambi<sub>g</sub>uit<sub>y</sub> arises as soon as $\tau _ { L }$ or $\tau _ { R }$ li<sub>es</sub> in<sub>s</sub>id<sub>e</sub> �<sub>,</sub> b<sub>ecause</sub> th<sub>e</sub> t<sub>rue</sub> $\tau _ { y }$ <sub>may s</sub>till f<sub>a</sub>ll <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> thi<sub>s</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>.</sub> $_ { \mathrm { A } s }$ ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub>d i<sub>n</sub> Figure 2, when $\tau _ { R }$ f<sub>a</sub>ll<sub>s</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l �<sub>,</sub> th<sub>e</sub> t<sub>rue</sub> $\tau _ { y }$ <sup>ma</sup>y <sup>occur</sup> at an<sub>y</sub> <sub>p</sub>oint <sub>w</sub>ithin $( \tau _ { R } , 1 )$ . Thus<sub>,</sub> we desi<sub>g</sub>n a wei<sub>g</sub>htin<sub>g</sub> strate<sub>gy</sub> f<sub>or</sub> dif<sub>eren</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>ons.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub> <sub>we</sub> <sub>use</sub> th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t $\tau _ { y }$ falls within � as their wei<sub>g</sub>hts and incor<sub>p</sub>orate these wei<sub>g</sub>hts i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on.</sub> H<sub>ence,</sub> th<sub>e</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f $N _ { k } ^ { D }$ <sub>un</sub>d<sub>er</sub> d<sub>ou</sub>bl<sub>y censore</sub>d d<sub>a</sub>t<sub>a</sub> i<sub>s g</sub>i<sub>ven</sub> b<sub>y,</sub>

$$
\begin{array} { c } { { N _ { k } ^ { D } = \displaystyle \sum _ { i = 1 } ^ { N } \Biggl \{ \underbrace { \mathbb { 1 } \left( \Delta _ { i } = 0 \right) \mathbb { 1 } \left( \tau _ { i y } \in I \right) } _ { \mathrm { u n c e u s o r e d } } } } \\ { { + \underbrace { \mathbb { 1 } \left( \Delta _ { i } = 1 \right) \left[ \displaystyle \frac { \tau _ { k + 1 } - \tau _ { i R } } { 1 - \tau _ { i R } } \mathbb { 1 } \left( \tau _ { i R } \in I \right) + \displaystyle \frac { \tau _ { k + 1 } - \tau _ { k } } { 1 - \tau _ { i R } } \mathbb { 1 } \left( \tau _ { i R } < \tau _ { k } \right) \right] } _ { \mathrm { r i g h t . c e n o r e d } } } } \\ { { + \underbrace { \mathbb { 1 } \left( \Delta _ { i } = 2 \right) \left[ \displaystyle \frac { \tau _ { i L } - \tau _ { k } } { \tau _ { i L } } \mathbb { 1 } \left( \tau _ { i L } \in I \right) + \displaystyle \frac { \tau _ { k + 1 } - \tau _ { k } } { \tau _ { i L } } \mathbb { 1 } \left( \tau _ { i L } > \tau _ { k + 1 } \right) \right] } _ { \mathrm { l e f t . c e n o r e d } } \Biggr \} } } \end{array}
$$

A<sub>ssume</sub> th<sub>e</sub> <sub>un</sub>it i<sub>n</sub>t<sub>erva</sub>l $[ 0 , 1 ]$ is divided into � non-overla<sub>pp</sub>in<sub>g</sub> <sub>su</sub>b<sub>-</sub>i<sub>n</sub>t<sub>erva</sub>l<sub>s.</sub> T<sub>o</sub> <sub>assess</sub> th<sub>e</sub> <sub>per</sub>f<sub>ormance</sub> <sub>across</sub> th<sub>ese</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s,</sub> <sub>we</sub> <sub>propose</sub> <sub>a</sub> <sub>me</sub>t<sub>r</sub>i<sub>c,</sub> E<sub>x</sub>t<sub>en</sub>d<sub>e</sub>d D<sub>-</sub>C<sub>a</sub>lib<sub>ra</sub>ti<sub>on,</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as:</sub>

$$
\mathrm { E x t e n d e d \ D \mathrm { - } C a l i b r a t i o n } = \sum _ { k = 1 } ^ { K - 1 } ( \tau _ { k + 1 } - \tau _ { k } - \frac { N _ { k } ^ { D } } { N } ) ^ { 2 }
$$

M<sub>oreover,</sub> f<sub>or</sub> th<sub>e propose</sub>d $N _ { k } ^ { D }$ <sub>,</sub> we <sub>p</sub>rove that the ex<sub>p</sub>ectation <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f $N _ { k } ^ { D } / N$ e<sub>q</sub>ua<sup>l</sup>s $\tau _ { k + 1 } - \tau _ { k }$ <sub>un</sub>d<sub>er</sub> <sub>some</sub> <sub>con</sub>diti<sub>ons,</sub> th<sub>ere</sub>b<sub>y</sub> <sub>p</sub>rovidin<sub>g</sub> a theoretical <sub>g</sub>uarantee for this wei<sub>g</sub>htin<sub>g</sub> scheme. This i<sub>s</sub> f<sub>orma</sub>li<sub>ze</sub>d i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>eorem:</sub>

Theorem 2. Under the independent censoring assumption, ifthe cumulative distribution function is strictly monotonically increasing, the expected proportion ofdoubly censored data in each bin equals the bin width, $i . e . , \mathbb { E } [ \frac { N _ { k } ^ { D } } { N } ] = \tau _ { k + 1 } - \tau _ { k }$

Proof<sub>.</sub> W<sub>e g</sub>i<sub>ve a p</sub>r<sub>oo</sub>f in A<sub>ppe</sub>ndix B<sub>.</sub>

## 5 Experiments

We conduct extensive ex<sub>p</sub>eriments to evaluate the <sub>p</sub>ro<sub>p</sub>osed metric <sub>an</sub>d <sub>a</sub>l<sub>gor</sub>ith<sub>m on</sub> b<sub>o</sub>th <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>an</sub>d <sub>rea</sub>l<sub>-wor</sub>ld <sub>re</sub>t<sub>a</sub>il d<sub>a</sub>t<sub>ase</sub>t<sub>s.</sub><sup>1</sup> Our t<sup>h</sup>ree primary researc<sup>h</sup> o<sup>b</sup>jectives are:

• How accuratel<sub>y</sub> does the <sub>p</sub>ro<sub>p</sub>osed metric assess model calib<sub>ra</sub>ti<sub>on un</sub>d<sub>er</sub> d<sub>ou</sub>bl<sub>y censore</sub>d d<sub>a</sub>t<sub>a compare</sub>d <sub>w</sub>ith <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> m<sub>easu</sub>r<sub>es</sub>?

• Does the <sub>p</sub>ro<sub>p</sub>osed method out<sub>p</sub>erform alternative a<sub>pp</sub>roaches i<sub>n ca</sub>lib<sub>ra</sub>ti<sub>on an</sub>d <sub>s</sub>h<sub>arpness</sub> i<sub>n</sub> b<sub>o</sub>th <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>an</sub>d <sub>rea</sub>l<sub>-wor</sub>ld <sub>sce</sub>n<sub>a</sub>ri<sub>os</sub>?

• Does the method translate into o<sub>p</sub>erational <sub>g</sub>ains, that is, su<sub>p</sub>erior inventor<sub>y</sub> mana<sub>g</sub>ement benefits?

Th<sub>ese</sub> th<sub>ree ques</sub>ti<sub>ons a</sub>dd<sub>ress</sub> th<sub>e</sub> di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>ve power an</sub>d <sub>ca</sub>lib<sub>ra-</sub> ti<sub>on accuracy o</sub>f <sub>our me</sub>t<sub>r</sub>i<sub>c,</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>ve e</sub>fi<sub>cacy o</sub>f <sub>our me</sub>th<sub>o</sub>d<sub>,</sub> and its <sub>p</sub>ractical utilit<sub>y</sub>. Because no <sub>g</sub>round-truth calibration mea-<sub>sure</sub> i<sub>s ava</sub>il<sub>a</sub>bl<sub>e on rea</sub>l d<sub>a</sub>t<sub>a, we</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>our me</sub>t<sub>r</sub>i<sub>c aga</sub>i<sub>ns</sub>t <sub>a</sub>lt<sub>erna</sub>ti<sub>ves</sub> i<sub>n syn</sub>th<sub>e</sub>ti<sub>c</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s.</sub> I<sub>n</sub> th<sub>ese</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s, mo</sub>d<sub>e</sub>l<sub>s are</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>so</sub>l<sub>e</sub>l<sub>y on censore</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons, w</sub>hil<sub>e</sub> t<sub>rue</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s are re-</sub> <sub>vea</sub>l<sub>e</sub>d <sub>on</sub>l<sub>y a</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> ti<sub>me</sub> t<sub>o compu</sub>t<sub>e</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>me</sub>t<sub>r</sub>i<sub>cs.</sub>

Re<sub>g</sub>ardin<sub>g</sub> im<sub>p</sub>lementation details<sub>,</sub> all models are im<sub>p</sub>lemented in P<sub>y</sub>T<sub>o</sub>r<sub>c</sub>h <sub>u</sub>tilizin<sub>g o</sub>n<sub>e se</sub>r<sub>ve</sub>r <sub>w</sub>ith <sub>a</sub>n Int<sub>e</sub>l X<sub>eo</sub>n CPU <sub>w</sub>ith 128 <sub>co</sub>r<sub>es,</sub> 228GB RAM <sub>a</sub>nd <sub>a</sub>n NVIDIA A100 GPU <sub>w</sub>ith 40GB m<sub>e</sub>m<sub>o</sub>r<sub>y.</sub> Th<sub>e op</sub>- timiz<sub>e</sub>r <sub>use</sub>d i<sub>s</sub> th<sub>e</sub> Ad<sub>a</sub>m <sub>op</sub>timiz<sub>e</sub>r <sub>w</sub>ith $\epsilon = 0 . 0 0 1$ <sub>, an</sub>d th<sub>e</sub> l<sub>earn</sub>i<sub>ng</sub> rate is selected from {0.001, 0.003, 0.006, 0.01} based on performance <sub>o</sub>n th<sub>e</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>o</sub>n <sub>se</sub>t<sub>.</sub> Aft<sub>e</sub>r <sub>so</sub>m<sub>e</sub> <sub>p</sub>r<sub>e</sub>limin<sub>a</sub>r<sub>y</sub> <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s,</sub> <sub>we</sub> fix<sub>e</sub>d � <sub>a</sub>t 20 f<sub>or a</sub>ll <sub>me</sub>th<sub>o</sub>d<sub>s</sub> t<sub>o ensure a</sub> f<sub>a</sub>i<sub>r compar</sub>i<sub>son.</sub>

## 5.1 Baselines

Gi<sub>ven</sub> th<sub>e</sub> l<sub>ac</sub>k <sub>o</sub>f <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>or</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng mo</sub>d<sub>e</sub>l<sub>s un</sub>d<sub>er</sub> d<sub>ou</sub>bl<sub>y</sub> <sub>censor</sub>i<sub>ng,</sub> <sub>we</sub> <sub>compare</sub> <sub>our</sub> <sub>approac</sub>h <sub>w</sub>ith <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d b<sub>ase-</sub> lines for <sub>q</sub>uantile modelin<sub>g</sub>: Excl. Censor[26], Lo<sub>g</sub>Norm MLE[26], S-CRPS[1], CQRNN[26].

## 5.2 Metrics

I<sub>n</sub> th<sub>e s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on s</sub>t<sub>u</sub>d<sub>y we</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>our me</sub>t<sub>r</sub>i<sub>c aga</sub>i<sub>ns</sub>t th<sub>ree</sub> widel<sub>y</sub> used calibration measures: ECE-uncensored, ECE[8, 9, 15, 33] and D-Calibration[16]. We take the oracle calibration com<sub>p</sub>uted f<sub>rom</sub> th<sub>e</sub> <sub>rea</sub>l l<sub>a</sub>b<sub>e</sub>l <sub>as</sub> th<sub>e</sub> <sub>go</sub>ld <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d<sub>.</sub> Wh<sub>en</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> DPE we additionall<sub>y</sub> re<sub>p</sub>ort shar<sub>p</sub>ness and CRPS[12] calculated on the t<sub>rue</sub> d<sub>eman</sub>d t<sub>o</sub> <sub>assess</sub> <sub>overa</sub>ll <sub>per</sub>f<sub>ormance.</sub> N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t <sub>s</sub>h<sub>arpness</sub> i<sub>s</sub> <sub>quan</sub>tifi<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>w</sub>idth <sub>o</sub>f th<sub>e</sub> 10th<sub>–</sub>90th <sub>quan</sub>til<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>an</sub>d <sub>we</sub> <sub>co</sub>m<sub>pu</sub>t<sub>e</sub> th<sub>e qua</sub>ntil<sub>e</sub>-b<sub>ase</sub>d CRPS in th<sub>e sa</sub>m<sub>e</sub> m<sub>a</sub>nn<sub>e</sub>r <sub>as</sub> B<sub>e</sub>rri<sub>sc</sub>h et al.[3].

## 5.3 Synthetic Datasets Results

5.3.1 Synthetic Datasets. Covariates and response. Coeficients <sub>are</sub> <sub>samp</sub>l<sub>e</sub>d <sub>once</sub> <sub>as</sub> $\beta _ { 0 } , \beta _ { 1 } , \beta _ { 2 } \sim \mathcal { U } ( 0 , 5 )$ <sub>an</sub>d $\beta _ { 3 } \sim \mathcal { U } ( 0 , 3 )$ . We th<sub>en</sub> d<sub>raw</sub> $x _ { 1 } \sim \mathcal { U } ( 0 , 3 ) , x _ { 2 } \sim \mathcal { U } ( 0 , 5 ) , x _ { 3 } \sim \mathcal { U } ( 0 , 5 )$ an<sup>d</sup> <sub>g</sub>enerate d<sub>eman</sub>d $y = \beta _ { 0 } + \beta _ { 1 } x _ { 1 } + \beta _ { 2 } \sin ( x _ { 2 } ) + \beta _ { 3 } \cos ( x _ { 3 } ) + \varepsilon ,$ <sub>, w</sub>h<sub>ere</sub> $\varepsilon \sim$ $N ( 0 , \sigma ^ { 2 } )$ <sub>.</sub> Th<sub>e s</sub>i<sub>nuso</sub>id<sub>a</sub>l t<sub>erms em</sub>b<sub>e</sub>d <sub>rea</sub>li<sub>s</sub>ti<sub>c non-</sub>li<sub>near</sub>it<sub>y, an</sub>d var<sub>y</sub>in<sub>g</sub> seeds allow us to simulate diverse censorin<sub>g</sub> scenarios.

Censoring mechanisms. We draw $L \sim \mathcal { U } ( 0 , 1 0 )$ <sub>an</sub>d <sub>se</sub>t $R =$ $L + \mathcal { U } ( 2 . 5 , 1 0 . 5 )$ <sub>;</sub> th<sub>e o</sub>b<sub>serve</sub>d l<sub>a</sub>b<sub>e</sub>l i<sub>s</sub> $y ^ { * } = \operatorname* { m a x } \{ L , \operatorname* { m i n } \{ y , R \} \}$

We <sub>g</sub>enerate 300<sub>,</sub>000 s<sub>y</sub>nthetic sam<sub>p</sub>les in each run and s<sub>p</sub>lit them into trainin<sub>g</sub> (60%), validation (20%) and test (20%) sets. All models <sub>were</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>on</sub> th<sub>e</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng se</sub>t <sub>us</sub>i<sub>ng censore</sub>d l<sub>a</sub>b<sub>e</sub>l<sub>s,</sub> th<sub>e</sub> b<sub>es</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>was se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>on</sub> th<sub>e va</sub>lid<sub>a</sub>ti<sub>on se</sub>t<sub>, an</sub>d th<sub>e</sub> fi<sub>na</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>was carr</sub>i<sub>e</sub>d <sub>ou</sub>t <sub>on</sub> th<sub>e</sub> t<sub>es</sub>t <sub>se</sub>t<sub>.</sub>

5.3.2 Metrics Comparison. We evaluate each calibration metric as a <sub>p</sub>rox<sub>y</sub> for true calibration <sub>q</sub>ualit<sub>y</sub>. For each simulation run<sub>,</sub> we rank a fixed set ofcandidate models (var<sub>y</sub>in<sub>g</sub> in architecture and re<sub>g</sub>- ularization) b<sub>y</sub> their oracle calibration error and inde<sub>p</sub>endentl<sub>y</sub> b<sub>y</sub> <sub>eac</sub>h <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>me</sub>t<sub>r</sub>i<sub>c.</sub> A <sub>me</sub>t<sub>r</sub>i<sub>c</sub> th<sub>a</sub>t <sub>preserves</sub> th<sub>e</sub> <sub>orac</sub>l<sub>e</sub>’<sub>s</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> i<sub>s</sub> <sub>cons</sub>id<sub>ere</sub>d <sub>we</sub>ll<sub>-a</sub>li<sub>gne</sub>d<sub>.</sub> F<sub>orma</sub>ll<sub>y,</sub> l<sub>e</sub>t $\mathbf { r } _ { \mathrm { o r a c l e } } \in \mathbb { R } ^ { M }$ <sub>an</sub>d $\mathbf { r } _ { \mathrm { m e t r i c } } \in \mathbb { R } ^ { M }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>ran</sub>k <sub>vec</sub>t<sub>ors</sub> <sub>over</sub> � <sub>mo</sub>d<sub>e</sub>l<sub>s;</sub> <sub>we</sub> <sub>compu</sub>t<sub>e</sub>

Kendall’s � [14] and S<sub>p</sub>earman’s $\rho \ [ 3 1 ]$ b<sub>e</sub>t<sub>ween</sub> th<sub>ese or</sub>d<sub>er</sub>i<sub>ngs.</sub> B<sub>o</sub>th <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> li<sub>e</sub> i<sub>n</sub> $[ - 1 , 1 ]$ <sub>,</sub> with values a<sub>pp</sub>roachin<sub>g</sub> 1 indicatin<sub>g</sub> stron<sub>g</sub> concordance and values near −1 indicatin<sub>g</sub> s<sub>y</sub>stematic in version. We avera<sub>g</sub>e coeficients across 500 inde<sub>p</sub>endent simulation <sub>runs;</sub> <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>s</sub>i<sub>gn</sub>ifi<sub>cance</sub> <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> <sub>nu</sub>ll h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> <sub>o</sub>f <sub>ran</sub>d<sub>om</sub> rankin<sub>g</sub> $( \tau = 0 )$ is assessed via the Wilcoxon si<sub>g</sub>ned-rank test [37].

Table 1: Performance comparison of calibration metrics. $^ { * * } ( p < 0 . 0 1 )$ (WSR vs Ext. D-Calibration).
<table><tr><td>Metrics</td><td>Kendall&#x27;s τ</td><td> $\mathbf { s p e a r m a n } ^ { \prime } s \rho$ </td></tr><tr><td>ECE</td><td> $- 0 . 2 2 0 ^ { * * }$ </td><td> $- 0 . 2 6 2 ^ { * * }$ </td></tr><tr><td>D-Calibration</td><td> $0 . 3 2 7 ^ { * * }$ </td><td> $0 . 4 1 4 ^ { * * }$ </td></tr><tr><td>ECE-uncensored</td><td> $0 . 4 8 6 ^ { * * }$ </td><td> $0 . 5 9 7 ^ { * * }$ </td></tr><tr><td>Ext. D-Calibration</td><td>0.740</td><td>0.842</td></tr></table>

Table 1 reveals a clear hierarchy in rank-alignment quality. Ext<sub>en</sub>d<sub>e</sub>d D<sub>-</sub>C<sub>a</sub>lib<sub>ra</sub>ti<sub>on ac</sub>hi<sub>eves</sub> th<sub>e s</sub>t<sub>ronges</sub>t <sub>concor</sub>d<sub>ance w</sub>ith th<sub>e</sub> oracle (Kendall’s $\tau = 0 . 7 4 0 ,$ <sub>,</sub> S<sub>p</sub>earman’s $\rho = 0 . 8 4 2 )$ , out<sub>p</sub>er<sup>f</sup>orm<sup>i</sup>n<sub>g</sub> the second-best metric<sub>,</sub> ECE-uncensored<sub>,</sub> b<sub>y</sub> substantial mar<sub>g</sub>ins (+52% and +41% in relative terms, res<sub>p</sub>ectivel<sub>y</sub>). Notabl<sub>y</sub>, standard ECE exhibits a negative correlation with true calibration quality $( \tau = - 0 . 2 2 0 )$ <sub>,</sub> confirmin<sub>g</sub> that in censored settin<sub>g</sub>s lower ECE values <sub>p</sub>aradoxicall<sub>y</sub> indicate worse calibration because ECE <sub>p</sub>enalizes <sub>pro</sub>b<sub>a</sub>bilit<sub>y mass ass</sub>i<sub>gne</sub>d b<sub>eyon</sub>d th<sub>e o</sub>b<sub>serve</sub>d h<sub>or</sub>i<sub>zon, w</sub>hi<sub>c</sub>h is actuall<sub>y</sub> a<sub>pp</sub>ro<sub>p</sub>riate for censored data. D-Calibration corrects for distributional assum<sub>p</sub>tions but remains limited $( \tau = 0 . 3 2 7 )$ while ECE-uncensored addresses censorin<sub>g y</sub>et i<sub>g</sub>nores the full <sub>p</sub>redi<sub>c</sub>ti<sub>ve</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>.</sub> Onl<sub>y</sub> Ext<sub>e</sub>nd<sub>e</sub>d D-C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n int<sub>eg</sub>r<sub>a</sub>t<sub>es</sub> b<sub>o</sub>th as<sub>p</sub>ects<sub>,</sub> <sub>y</sub>ieldin<sub>g</sub> a statisticall<sub>y</sub> si<sub>g</sub>nificant im<sub>p</sub>rovement over all b<sub>ase</sub>li<sub>nes</sub> $( p < 0 . 0 1$ , Wilcoxon si<sub>g</sub>ned-rank test) and a<sub>pp</sub>roachin<sub>g</sub> th<sub>e</sub> <sub>re</sub>li<sub>a</sub>bilit<sub>y</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>requ</sub>i<sub>re</sub>d f<sub>or</sub> <sub>mo</sub>d<sub>e</sub>l <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> i<sub>n</sub> <sub>opera</sub>ti<sub>ona</sub>l <sup>d</sup>e<sub>p</sub><sup>l</sup>o<sub>y</sub>ments.

5.3.3 Loss Comparison. To evaluate DPE against competing loss f<sub>unc</sub>ti<sub>ons, we</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>w</sub>ith E<sub>x</sub>t<sub>en</sub>d<sub>e</sub>d D<sub>-</sub>C<sub>a</sub>lib<sub>ra</sub>ti<sub>on, s</sub>h<sub>arpness,</sub> oracle calibration, and real CRPS (com<sub>p</sub>uted on the real labels). As in the metrics com<sub>p</sub>arison<sub>,</sub> results are avera<sub>g</sub>ed across simulation runs and reported in Table 2, with significance assessed via the Wilcoxon si<sub>g</sub>ned-rank test usin<sub>g</sub> the same <sub>p</sub>rotocol.

As reported in Table 2, DPE achieves the minimum value in almost all metrics and si<sub>g</sub>nificantl<sub>y</sub> exceeds all baselines (Wilcoxon <sub>s</sub>i<sub>gne</sub>d <sub>ran</sub>k t<sub>es</sub>t<sub>,</sub> $\begin{array} { r } { p < 0 . 0 1 ) } \end{array}$ . S<sub>p</sub>ecificall<sub>y,</sub> its Extended D-Calibration <sub>excee</sub>d<sub>s</sub> th<sub>e secon</sub>d<sub>-</sub>b<sub>es</sub>t <sub>resu</sub>lt b<sub>y</sub> 44<sub>.</sub>16%<sub>, w</sub>hil<sub>e</sub> th<sub>e rea</sub>l CRPS i<sub>m-</sub> <sub>proves</sub> b<sub>y</sub> 6<sub>.</sub>52%<sub>,</sub> <sub>con</sub>fi<sub>rm</sub>i<sub>ng</sub> b<sub>o</sub>th th<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>ve</sub> <sub>a</sub>d<sub>van</sub>t<sub>ages</sub> <sub>o</sub>f th<sub>e</sub> <sub>propose</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>or.</sub>

5.3.4 InventorySimulation. We examine practical implementations <sub>an</sub>d <sub>assess</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>veness o</sub>f <sub>var</sub>i<sub>ous approac</sub>h<sub>es</sub> i<sub>n rea</sub>l<sub>-wor</sub>ld <sub>sce-</sub> <sub>nar</sub>i<sub>os.</sub> F<sub>or</sub> l<sub>arge-sca</sub>l<sub>e</sub> i<sub>nven</sub>t<sub>ory managemen</sub>t<sub>,</sub> th<sub>e s</sub>h<sub>eer num</sub>b<sub>er o</sub>f SKUs makes manual stockin<sub>g</sub> extremel<sub>y</sub> labor-intensive and costl<sub>y</sub>. I<sub>n</sub> <sub>con</sub>t<sub>ras</sub>t<sub>,</sub> <sub>mo</sub>d<sub>e</sub>l<sub>-</sub>b<sub>ase</sub>d <sub>s</sub>t<sub>oc</sub>ki<sub>ng</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>o</sub>f<sub>er</sub> <sub>cos</sub>t <sub>e</sub>fi<sub>c</sub>i<sub>ency</sub> <sub>an</sub>d <sub>more</sub> <sub>accura</sub>t<sub>e</sub> d<sub>eman</sub>d f<sub>orecas</sub>ti<sub>ng.</sub>

Given the numerous SKUs involved<sub>,</sub> we t<sub>yp</sub>icall<sub>y</sub> re<sub>g</sub>ulate the overall stockin<sub>g</sub> <sub>q</sub>uantit<sub>y</sub> of all SKUs in this cate<sub>g</sub>or<sub>y</sub> to maximize GMV<sub>.</sub> In <sub>ou</sub>r <sub>s</sub>im<sub>u</sub>l<sub>a</sub>ti<sub>o</sub>n<sub>,</sub> <sub>we</sub> <sub>assu</sub>m<sub>e</sub> th<sub>e</sub>r<sub>e</sub> <sub>a</sub>r<sub>e</sub> N SKU<sub>s,</sub> <sub>eac</sub>h <sub>w</sub>ith dif<sub>eren</sub>t <sub>s</sub>t<sub>oc</sub>k<sub>ou</sub>t <sub>cos</sub>t<sub>s</sub> $c _ { 1 }$ <sub>an</sub>d <sub>overs</sub>t<sub>oc</sub>k <sub>cos</sub>t<sub>s</sub> $c _ { 2 } ,$ b<sub>o</sub>th f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>a u</sub>nif<sub>o</sub>rm di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n $\mathrm { U } ( 0 , 1 )$ . We aim to maximize the followin<sub>g</sub> ex<sub>p</sub>ectat<sup>i</sup>on:

$$
\mathbb { E } _ { y \sim p ( y | x ) } { \left\{ y - { \left[ c _ { 1 } ( y - t ) ^ { + } + c _ { 2 } ( t - y ) ^ { + } \right] } \right\} }
$$

<sub>w</sub>h<sub>ere</sub> � i<sub>s es</sub>ti<sub>ma</sub>t<sub>e</sub>d d<sub>eman</sub>d <sub>an</sub>d <sub>�</sub> i<sub>s</sub> th<sub>e</sub> d<sub>eman</sub>d<sub>.</sub> W<sub>e can ana</sub>l<sub>y</sub>ti<sub>ca</sub>l d<sub>er</sub>i<sub>ve</sub> th<sub>e a</sub>b<sub>ove expec</sub>t<sub>a</sub>ti<sub>on an</sub>d <sub>ge</sub>t

$$
\begin{array} { r l } & { \mathbb { E } _ { y \sim p ( y \vert x ) } \{ y - [ c _ { 1 } ( { \bar { y } } - t ) ^ { + } + c _ { 2 } ( t - y ) ^ { + } ] \} } \\ & { = \underbrace { \mathbb { E } ( { \mathcal { Y } } \vert { \mathcal { X } } ) } _ { \mathrm { E x p e c t a t i o n } } - \underbrace { \{ \int _ { t } ^ { \infty } c _ { 1 } ( y - t ) p ( y \vert x ) d y } _ { \mathrm { S t o c k o u t l o s s } } + \underbrace { \int _ { 0 } ^ { t } c _ { 2 } ( t - y ) p ( y \vert x ) d y } _ { \mathrm { O v e r s t o c k l o s s } } \} } \end{array}
$$

Th<sub>e s</sub>t<sub>oc</sub>k<sub>ou</sub>t l<sub>oss an</sub>d <sub>overs</sub>t<sub>oc</sub>k l<sub>oss can</sub> b<sub>e</sub> f<sub>ur</sub>th<sub>er</sub> d<sub>er</sub>i<sub>ve</sub>d <sub>as</sub>

$$
\begin{array} { l } { { L _ { 1 } ( q ) = \displaystyle \int _ { t } ^ { \infty } c _ { 1 } ( y - t ) \rho ( y | x ) d y } } \\ { { \mathrm { } } } \\ { { \displaystyle = c _ { 1 } [ \sum _ { q = q _ { t } } ^ { 1 } \hat { F } ^ { - 1 } ( q | x ) \cdot \Delta q - \hat { F } ^ { - 1 } ( q _ { t } | x ) ( 1 - q _ { t } ) ] } } \\ { { \displaystyle \qquad } } \\ { { \displaystyle L _ { 2 } ( q ) = \int _ { 0 } ^ { t } c _ { 2 } ( t - y ) \rho ( y ) d y } } \\ { { \mathrm { } } } \\ { { \displaystyle = c _ { 2 } [ q _ { t } \cdot \hat { F } ^ { - 1 } ( q _ { t } | x ) - \sum _ { q = 0 } ^ { q _ { t } } \hat { F } ^ { - 1 } ( q | x ) \cdot \Delta q ] } } \end{array}
$$

Wh<sub>ere</sub> ${ \hat { F } } ^ { - 1 }$ i<sub>s</sub> th<sub>e es</sub>tim<sub>a</sub>t<sub>e o</sub>f th<sub>e</sub> in<sub>ve</sub>r<sub>se</sub> CDF<sub>,</sub> $q _ { t } \ = \ F ( t | x )$ <sub>an</sub>d $\Delta q = 0 . 0 1$ . The detailed deri<sub>v</sub>ation is <sub>g</sub>i<sub>v</sub>en in A<sub>pp</sub>endix C.

Gi<sub>v</sub>en a set of<sub>qu</sub>antiles $\{ \hat { F } ^ { - 1 } ( q _ { i 1 } | x _ { i } ) , \hat { F } ^ { - 1 } ( q _ { i 2 } | x _ { i } ) , . . . , \hat { F } ^ { - 1 } ( q _ { i K } | x _ { i } ) \} _ { i = 1 } ^ { N }$ for N SKUs<sub>,</sub> we o<sub>p</sub>timize the followin<sub>g</sub> inte<sub>g</sub>er <sub>p</sub>ro<sub>g</sub>rammin<sub>g</sub> <sub>p</sub>robl<sub>em:</sub>

$$
\begin{array} { r l } { M a x i m i z e } & { \displaystyle \sum _ { i = 1 } ^ { N } \displaystyle \sum _ { j = 1 } ^ { K } \mathbb { E } ( y _ { i } | X _ { i } ) - \left[ L _ { 1 } ( q _ { i j } ) + L _ { 2 } ( q _ { i j } ) \right] \cdot z _ { i j } } \\ { \mathrm { s u b j e c t ~ t o ~ } } & { \displaystyle \sum _ { j = 1 } ^ { K } z _ { i j } = 1 , \quad i = 1 , 2 , \ldots , N } \\ & { \displaystyle \sum _ { i = 1 } ^ { N } \displaystyle \sum _ { j = 1 } ^ { K } z _ { i j } \cdot \hat { F } ^ { - 1 } ( q _ { i j } ) \leq T \cdot ( 1 + \alpha ) } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $T$ <sub>represen</sub>t<sub>s expec</sub>t<sub>e</sub>d t<sub>o</sub>t<sub>a</sub>l <sub>vo</sub>l<sub>ume,</sub> $\alpha \in \left[ 0 , 0 . 4 \right]$ serves as a <sub>vo</sub>l<sub>ume con</sub>t<sub>ro</sub>l <sub>parame</sub>t<sub>er an</sub>d $F ^ { - 1 } ( q _ { i j } )$ i<sub>s roun</sub>d<sub>e</sub>d t<sub>o</sub> th<sub>e neares</sub>t <sup>i</sup>nte<sub>g</sub>er.

Experimental Setup. We simulate inventory dynamics for $N =$ 200 SKUs. We evaluate methods across a <sub>g</sub>rid of service-level tar<sub>g</sub>ets<sub>,</sub> <sup>var</sup>y<sup>in</sup>g $\alpha \in \{ 0 , 0 . 0 5 , 0 . 1 0 , \ldots , 0 . 4 0 \}$ <sub>,</sub> where hi<sub>g</sub>her <sub>� p</sub>rioritizes <sub>s</sub>t<sub>oc</sub>k<sub>ou</sub>t r<sub>e</sub>d<sub>uc</sub>ti<sub>o</sub>n <sub>ove</sub>r <sub>ove</sub>r<sub>s</sub>t<sub>oc</sub>kin<sub>g</sub> miti<sub>ga</sub>ti<sub>o</sub>n<sub>.</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> i<sub>s</sub> measured by two operational metrics: (i) overstocking rate, defined as excess inventory divided by total sales; and (ii) stockout SKU ratio, defined as the proportion of SKUs that experience a complete <sub>se</sub>ll<sub>ou</sub>t d<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on.</sub>

Findings. Figure 3 reveals a clear hierarchical ordering of method performance. DPE achieves the minimal stockout SKU ratio for any fixed overstocking level, and conversely, the minimal overstocking rate for any fixed stockout threshold. This reciprocal dominance indi<sub>ca</sub>t<sub>es</sub> th<sub>a</sub>t DPE’<sub>s</sub> P<sub>a</sub>r<sub>e</sub>t<sub>o</sub> fr<sub>o</sub>nti<sub>e</sub>r <sub>s</sub>tri<sub>c</sub>tl<sub>y</sub> d<sub>o</sub>min<sub>a</sub>t<sub>es</sub> th<sub>ose o</sub>f CQRNN, Excl. Censor, and S-CRPS across all operational regimes. For exam<sub>p</sub>le<sub>,</sub> at an overstockin<sub>g</sub> rate of 0.25<sub>,</sub> DPE reduces the stockout ratio to 0.23 com<sub>p</sub>ared to 0.25 for S-CRPS (the second-best method), a relative im<sub>p</sub>rovement of 8%. These <sub>g</sub>ains underscore the o<sub>p</sub>erational value of DPE in retail inventor<sub>y</sub> control<sub>,</sub> <sub>p</sub>articularl<sub>y</sub> i<sub>n</sub> <sub>se</sub>tti<sub>ngs</sub> <sub>w</sub>h<sub>ere</sub> b<sub>o</sub>th h<sub>o</sub>ldi<sub>ng</sub> <sub>cos</sub>t<sub>s</sub> <sub>an</sub>d l<sub>os</sub>t<sub>-sa</sub>l<sub>es</sub> <sub>pena</sub>lti<sub>es</sub> <sub>are</sub> economicall<sub>y</sub> si<sub>g</sub>nificant.

Table 2: Performance metrics for diferent survival methods. Bold = best; underline = runner-up; $^ { \star } p < 0 . 0 5 ,$ <sup>★★</sup>� < 0.01 vs DPE (WSR, except Sharpness).
<table><tr><td>Model</td><td>Extended D-Calibration</td><td>Oracle Calibration</td><td>Sharpness</td><td>Real CRPS</td></tr><tr><td>Excl. Censor</td><td> $0 . 1 5 4 ^ { \star \star }$ </td><td> $0 . 4 9 2 ^ { \star \star }$ </td><td>8.646</td><td> $0 . 0 9 3 ^ { \star \star }$ </td></tr><tr><td>LogNorm</td><td> $\overline { { 0 . 8 3 2 ^ { \star \star } } }$ </td><td> $1 . 7 6 1 ^ { \star \star }$ </td><td>12.101</td><td> $0 . 1 6 4 ^ { \star \star }$ </td></tr><tr><td>CQRNN</td><td> $0 . 2 9 3 ^ { \star \star }$ </td><td> $0 . 5 8 5 ^ { \star \star }$ </td><td>9.039</td><td> $0 . 1 1 4 ^ { \star \star }$ </td></tr><tr><td>S-CRPS</td><td> $0 . 1 8 9 ^ { \star \star }$ </td><td> $0 . 5 1 3 ^ { \star \star }$ </td><td>9.296</td><td> $0 . 0 9 2 ^ { \star \star }$ </td></tr><tr><td>DPE (Our Method)</td><td>0.086</td><td>0.486</td><td>11.218</td><td>0.086</td></tr></table>

Table 3: Calibration and sharpness comparison (lower is better).
<table><tr><td>Dataset</td><td colspan="2">Youxuan</td><td colspan="2">Maicai</td></tr><tr><td>Metrics</td><td>Extended D-Calibration</td><td>Sharpness</td><td>Extended D-Calibration</td><td>Sharpness</td></tr><tr><td>Excl. censor</td><td>0.049</td><td>76.72</td><td>0.17</td><td>9.82</td></tr><tr><td>LogNorm</td><td>0.039</td><td>107.96</td><td>0.14</td><td>10.8</td></tr><tr><td>S-CRPS</td><td>0.045</td><td>77.54</td><td>0.13</td><td>11.67</td></tr><tr><td>CQRNN</td><td>0.042</td><td>77.67</td><td>0.07</td><td>10.45</td></tr><tr><td>DPE</td><td>0.040</td><td>76.4</td><td>0.05</td><td>10.3</td></tr></table>

![](images/7f2f5da57460f57af6472baa037e80d381167324a90b93cdedfba114e9819d64.jpg)  
Figure 3: Inventory Simulation.

## 5.4 Real-world Datasets Results

5.4.1 Real-world Datasets. For real-world experiments, since publi<sub>c re</sub>t<sub>a</sub>il d<sub>a</sub>t<sub>ase</sub>t<sub>s suc</sub>h <sub>as</sub> M5 <sub>an</sub>d F<sub>avor</sub>it<sub>a</sub> d<sub>o no</sub>t <sub>co</sub>ll<sub>ec</sub>t <sub>censor</sub>i<sub>ng</sub> information<sub>,</sub> which is <sub>p</sub>revalent in e-commerce retailin<sub>g,</sub> we coll<sub>ec</sub>t<sub>e</sub>d t<sub>wo</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> f<sub>rom</sub> t<sub>wo on</sub>li<sub>ne grocery re</sub>t<sub>a</sub>ili<sub>ng p</sub>l<sub>a</sub>tf<sub>orms o</sub>f Meit<sub>u</sub>an<sub>,</sub> namel<sub>y</sub> Maicai and Yo<sub>u</sub>x<sub>u</sub>an<sub>,</sub> co<sub>v</sub>erin<sub>g</sub> the <sub>p</sub>eriod from October 10th to No<sub>v</sub>ember 15th<sub>,</sub> 2023. Notabl<sub>y,</sub> Maicai and Yo<sub>u</sub>x<sub>u</sub>an h<sub>ave</sub> dif<sub>eren</sub>t b<sub>us</sub>i<sub>ness mo</sub>d<sub>e</sub>l<sub>s, resu</sub>lti<sub>ng</sub> i<sub>n</sub> di<sub>s</sub>ti<sub>nc</sub>tl<sub>y</sub> dif<sub>eren</sub>t <sub>sa</sub>l<sub>es</sub> distributions. We aim to demonstrate the <sub>g</sub>eneralizabilit<sub>y</sub> of our <sub>me</sub>th<sub>o</sub>d b<sub>y</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> it<sub>s</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>veness</sub> <sub>across</sub> th<sub>ese</sub> dif<sub>eren</sub>t t<sub>ypes</sub> of distributions. Each dataset is s<sub>p</sub>lit into three subsets: the initial

30 d<sub>ays</sub> <sub>are</sub> <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub>d f<sub>or</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> 1 d<sub>ay</sub> f<sub>or</sub> <sub>va</sub>lid<sub>a-</sub> ti<sub>on,</sub> <sub>an</sub>d th<sub>e</sub> fi<sub>na</sub>l 4 d<sub>ays</sub> <sub>are</sub> <sub>reserve</sub>d f<sub>or</sub> t<sub>es</sub>ti<sub>ng.</sub> Si<sub>nce</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> are <sub>p</sub>ro<sub>p</sub>rietar<sub>y</sub> due to data <sub>p</sub>rivac<sub>y</sub> concerns<sub>,</sub> we <sub>p</sub>rovide detailed descri<sub>p</sub>tions of our datasets to ensure re<sub>p</sub>roducibilit<sub>y</sub>.

• Youxuan (community group-buying platform): 8.52 million <sub>recor</sub>d<sub>s;</sub> 19<sub>.</sub>84% <sub>r</sub>i<sub>g</sub>ht<sub>-censore</sub>d<sub>,</sub> 3<sub>.</sub>56% l<sub>e</sub>ft<sub>-censore</sub>d<sub>.</sub>

• Maicai (online supermarket platform): 18.07 million records; 18<sub>.</sub>52% <sub>r</sub>i<sub>g</sub>ht<sub>-censore</sub>d<sub>,</sub> 6<sub>.</sub>26% l<sub>e</sub>ft<sub>-censore</sub>d<sub>.</sub>

C<sub>ensor</sub>i<sub>ng</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s are</sub> d<sub>er</sub>i<sub>ve</sub>d f<sub>rom sys</sub>t<sub>em</sub> l<sub>ogs.</sub> St<sub>oc</sub>k<sub>ou</sub>t<sub>s are</sub> <sub>r</sub>i<sub>g</sub>ht<sub>-censore</sub>d $( \Delta = 1 , R = { \mathrm { s a l e s } } )$ <sub>.</sub> P<sub>romo</sub>ti<sub>ons are reac</sub>ti<sub>ve an</sub>d <sub>unp</sub>l<sub>anne</sub>d<sub>;</sub> th<sub>e resu</sub>lti<sub>ng sa</sub>l<sub>es surge</sub> i<sub>s</sub> t<sub>rea</sub>t<sub>e</sub>d <sub>as</sub> l<sub>e</sub>ft<sub>-censore</sub>d $( \Delta = 2 , L = \mathrm { s a l e s } )$ <sub>, w</sub>ith th<sub>e</sub> t<sub>arge</sub>t b<sub>e</sub>i<sub>ng</sub> b<sub>ase</sub>li<sub>ne</sub> d<sub>eman</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>promo</sub>ti<sub>on.</sub> N<sub>orma</sub>l <sub>sa</sub>l<sub>es are uncensore</sub>d $( \Delta = 0 , L = R = \mathrm { s a l e s } )$

B<sub>o</sub>th d<sub>a</sub>t<sub>ase</sub>t<sub>s s</sub>h<sub>are</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> f<sub>ea</sub>t<sub>ures:</sub> t<sub>ra</sub>fi<sub>c, sa</sub>l<sub>es, an</sub>d <sub>pr</sub>i<sub>ce</sub> histories. The data are <sub>p</sub>artitioned as follows: trainin<sub>g</sub> on da<sub>y</sub>s 1–30<sub>,</sub> validation on da<sub>y</sub> 31<sub>,</sub> and testin<sub>g</sub> on da<sub>y</sub>s 32–35.

5.4.2 Loss Comparison. Empirical results on synthetic data confi<sub>rm</sub> th<sub>a</sub>t E<sub>x</sub>t<sub>en</sub>d<sub>e</sub>d D<sub>-</sub>C<sub>a</sub>lib<sub>ra</sub>ti<sub>on a</sub>li<sub>gns mos</sub>t <sub>c</sub>l<sub>ose</sub>l<sub>y w</sub>ith th<sub>e orac</sub>l<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>ti<sub>on score.</sub> O<sub>n</sub> th<sub>e</sub> t<sub>wo rea</sub>l<sub>-wor</sub>ld d<sub>a</sub>t<sub>ase</sub>t<sub>s, w</sub>h<sub>ere</sub> t<sub>rue</sub> d<sub>e-</sub> <sub>man</sub>d i<sub>s unava</sub>il<sub>a</sub>bl<sub>e</sub> f<sub>or censore</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons, we ran</sub>k <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> solely by Extended D-Calibration and sharpness. As shown in Table 3, DPE ranks in the top two for every metric, demonstrating its <sub>e</sub>f<sub>ec</sub>ti<sub>veness.</sub>

On Y<sub>ou</sub>x<sub>ua</sub>n<sub>,</sub> DPE r<sub>a</sub>nk<sub>s seco</sub>nd f<sub>o</sub>r b<sub>o</sub>th m<sub>e</sub>tri<sub>cs.</sub> L<sub>og</sub>N<sub>o</sub>rm <sub>a</sub>ttains a mar<sub>g</sub>inall<sub>y</sub> lower Extended D-Calibration (0.039 vs. 0.040), <sub>y</sub>et its shar<sub>p</sub>ness is 41.3% worse than that of DPE (107.96 vs. 76.4). N<sub>o</sub>t<sub>a</sub>bl<sub>y,</sub> L<sub>og</sub>N<sub>orm</sub> <sub>ac</sub>hi<sub>eves</sub> th<sub>e</sub> b<sub>es</sub>t <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> b<sub>u</sub>t th<sub>e</sub> <sub>wors</sub>t <sub>s</sub>h<sub>arp-</sub> <sub>ness among a</sub>ll <sub>mo</sub>d<sub>e</sub>l<sub>s, un</sub>d<sub>erscor</sub>i<sub>ng</sub> th<sub>a</sub>t <sub>ca</sub>lib<sub>ra</sub>ti<sub>on a</sub>l<sub>one</sub> i<sub>s</sub> i<sub>nsu</sub>f<sub>-</sub> fi<sub>c</sub>i<sub>en</sub>t f<sub>or mo</sub>d<sub>e</sub>l <sub>compar</sub>i<sub>son.</sub>

On M<sub>a</sub>i<sub>ca</sub>i<sub>,</sub> DPE <sub>ac</sub>hi<sub>eves</sub> th<sub>e</sub> l<sub>owes</sub>t Ext<sub>e</sub>nd<sub>e</sub>d D-C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n (0.05), out<sub>p</sub>erformin<sub>g</sub> the second-best method (CQRNN, 0.07) b<sub>y</sub> 28<sub>.</sub>6%<sub>.</sub> R<sub>e</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> Ex<sub>c</sub>l<sub>.</sub> C<sub>e</sub>n<sub>so</sub>r<sub>,</sub> it tr<sub>a</sub>d<sub>es</sub> <sub>a</sub> m<sub>o</sub>d<sub>es</sub>t <sub>wo</sub>r<sub>se</sub>nin<sub>g</sub> in shar<sub>p</sub>ness (10.3 vs. 9.82) for a 70.6% reduction in calibration error (0.05 vs. 0.17). Notabl<sub>y</sub>, Excl. Censor, des<sub>p</sub>ite exhibitin<sub>g</sub> the hi<sub>g</sub>hest s<sup>h</sup>arpness, ran<sup>k</sup>s worst in ca<sup>l</sup>i<sup>b</sup>ration, rein<sup>f</sup>orcing t<sup>h</sup>e nee<sup>d</sup> to joint<sup>l</sup>y <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> b<sub>o</sub>th <sub>cr</sub>it<sub>er</sub>i<sub>a.</sub>

5.4.3 Inventory Online A/B Test. For the A/B test, we randomly <sub>ass</sub>i<sub>gn</sub> <sub>pro</sub>d<sub>uc</sub>t<sub>s</sub> f<sub>rom</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>ware</sub>h<sub>ouses,</sub> <sub>across</sub> dif<sub>eren</sub>t <sub>su</sub>b<sub>-</sub> cate<sub>g</sub>ories<sub>,</sub> to treatment and control <sub>g</sub>rou<sub>p</sub>s with e<sub>q</sub>ual <sub>p</sub>robabilit<sub>y</sub>. Gi<sub>ven</sub> th<sub>e</sub> t<sub>ra</sub>d<sub>e-o</sub>f b<sub>e</sub>t<sub>ween overs</sub>t<sub>oc</sub>ki<sub>ng ra</sub>t<sub>e an</sub>d <sub>s</sub>t<sub>oc</sub>k<sub>ou</sub>t SKU <sub>ra</sub>ti<sub>o, we co</sub>ll<sub>ec</sub>t d<sub>a</sub>t<sub>a on</sub> b<sub>o</sub>th <sub>me</sub>t<sub>r</sub>i<sub>cs over a</sub> 7<sub>-</sub>d<sub>ay per</sub>i<sub>o</sub>d<sub>.</sub> Th<sub>e</sub> <sub>compar</sub>i<sub>son s</sub>h<sub>ows</sub> th<sub>a</sub>t<sub>,</sub> f<sub>or a</sub> fi<sub>xe</sub>d <sub>s</sub>t<sub>oc</sub>k<sub>ou</sub>t SKU <sub>ra</sub>ti<sub>o, our mo</sub>d<sub>e</sub>l <sub>re</sub>d<sub>uces</sub> th<sub>e overs</sub>t<sub>oc</sub>k <sub>ra</sub>t<sub>e</sub> b<sub>y</sub> 8% <sub>compare</sub>d t<sub>o</sub> th<sub>e</sub> i<sub>ncum</sub>b<sub>en</sub>t <sub>on</sub>li<sub>ne</sub> al<sub>g</sub>orithm<sub>,</sub> demonstratin<sub>g</sub> its <sub>p</sub>ractical value in <sub>p</sub>roduction environments.

## 6 Conclusion

Thi<sub>s s</sub>t<sub>u</sub>d<sub>y</sub> d<sub>eve</sub>l<sub>ops a nove</sub>l <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> f<sub>ramewor</sub>k th<sub>a</sub>t <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> <sub>a</sub>dd<sub>resses</sub> d<sub>ou</sub>bl<sub>y censore</sub>d d<sub>a</sub>t<sub>a preva</sub>l<sub>en</sub>t i<sub>n e-commerce</sub> d<sub>eman</sub>d forecastin<sub>g</sub>. S<sub>p</sub>ecificall<sub>y,</sub> we <sub>p</sub>ro<sub>p</sub>ose a dee<sub>p</sub> learnin<sub>g</sub>-based <sub>q</sub>uantile re<sub>g</sub>ression framework, desi<sub>g</sub>n the Doubl<sub>y</sub> Portno<sub>y</sub> Estimator (DPE), an iterative re-wei<sub>g</sub>htin<sub>g</sub> al<sub>g</sub>orithm for trainin<sub>g</sub> dee<sub>p</sub> models with <sub>censore</sub>d d<sub>a</sub>t<sub>a, an</sub>d d<sub>eve</sub>l<sub>op</sub> E<sub>x</sub>t<sub>en</sub>d<sub>e</sub>d D<sub>-</sub>C<sub>a</sub>lib<sub>ra</sub>ti<sub>on, a me</sub>t<sub>r</sub>i<sub>c</sub> t<sub>o</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e mo</sub>d<sub>e</sub>l<sub>s un</sub>d<sub>er</sub> d<sub>ou</sub>bl<sub>e censor</sub>i<sub>ng.</sub> Th<sub>e e</sub>fi<sub>cacy o</sub>f <sub>our</sub> f<sub>rame</sub> <sub>wor</sub>k i<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub>d <sub>v</sub>i<sub>a s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on an</sub>d t<sub>wo rea</sub>l<sub>-wor</sub>ld d<sub>a</sub>t<sub>ase</sub>t<sub>s.</sub> A<sub>pp</sub>lication res<sub>u</sub>lts <sub>v</sub>erif<sub>y</sub> that the <sub>p</sub>ro<sub>p</sub>osed method efecti<sub>v</sub>el<sub>y</sub> miti<sub>g</sub>ates overstockin<sub>g</sub> and stockout losses in inventor<sub>y</sub> mana<sub>g</sub>ement<sub>,</sub> thereb<sub>y</sub> im<sub>p</sub>rovin<sub>g</sub> o<sub>p</sub>erational eficienc<sub>y</sub> and <sub>p</sub>rofitabilit<sub>y</sub>.

## 7 Appendices

## 7.1 Proof of Theorem 1

We <sub>p</sub>rove that the Doubl<sub>y</sub> Portno<sub>y</sub>’s Estimator (DPE) is a <sub>p</sub>ro<sub>p</sub>er scorin<sub>g</sub> r<sub>u</sub>le<sub>,</sub> i.e.<sub>,</sub>

$$
\mathbb { E } _ { y } [ S _ { D P E } ( y ^ { * } , \hat { y } _ { \tau } , w _ { l } , w _ { r } , \tau ) ] \ge \mathbb { E } _ { y } [ S _ { D P E } ( y ^ { * } , y _ { \tau } , w _ { l } , w _ { r } , \tau ) ]\tag{1}
$$

f<sub>or a</sub>ll $\hat { y } _ { \tau }$ <sub>,</sub> with e<sub>q</sub>ualit<sub>y</sub> if $\hat { y } _ { \tau } = y _ { \tau } = F ^ { - 1 } ( \tau | x )$

The <sub>p</sub>roof <sub>p</sub>roceeds b<sub>y</sub> the relati<sub>v</sub>e <sub>p</sub>ositions of $\hat { y } _ { \tau }$ <sub>an</sub>d $y _ { \tau }$ to the censorin<sub>g</sub> bounds [�, �]. We <sub>p</sub>artition into three structurall<sub>y</sub> di<sub>s</sub>tin<sub>c</sub>t <sub>cases;</sub> <sub>sce</sub>n<sub>a</sub>ri<sub>os</sub> <sub>w</sub>ithin <sub>eac</sub>h <sub>case</sub> <sub>s</sub>h<sub>a</sub>r<sub>e</sub> th<sub>e</sub> <sub>sa</sub>m<sub>e</sub> d<sub>e</sub>ri<sub>va</sub>ti<sub>o</sub>n p<sup>attern.</sup>

(1) $m a x \{ \hat { y } _ { \tau } , y _ { \tau } \} < L$

(2) $\hat { y } _ { \tau } < L , L \leq y _ { \tau } \leq R$

(3) $\hat { y } _ { \tau } < L , y _ { \tau } > R$

(4) $L \leq \hat { y } _ { \tau } \leq R , y _ { \tau } < L$

(5) $L \leq \hat { y } _ { \tau } \leq R , L \leq y _ { \tau } \leq R$

(6) $L \leq \hat { y } _ { \tau } \leq R , y _ { \tau } > R$

(7) $\hat { y } _ { \tau } > R , y _ { \tau } < L$

(8) $\hat { y } _ { \tau } > R , L \leq y _ { \tau } \leq R$

(9) $m i n \{ y _ { \tau } , \hat { y } _ { \tau } \} > R$

7.1.1 Case I: Both Predictions on the Same Side. This covers Sce-<sub>nar</sub>i<sub>os</sub> 1 <sub>an</sub>d 9<sub>: max</sub> $\{ \hat { y } _ { \tau } , y _ { \tau } \} < L$ <sub>o</sub>r min $\{ \hat { y } _ { \tau } , y _ { \tau } \} > R$

Scenario 1: max $\{ \hat { y } _ { \tau } , y _ { \tau } \} < L$ . The DPE loss takes the form:

$$
\begin{array} { r l } & { S _ { D P E } ( y ^ { * } , \hat { y } _ { \tau } , w _ { l } , w _ { r } , \tau ) } \\ & { = \left\{ \begin{array} { c c } { ( 1 - w _ { l } ) ( z _ { - \infty } - \hat { y } _ { \tau } ) ( \tau - 1 ) + w _ { l } ( L - \hat { y } _ { \tau } ) \tau , } & { y < L , } \\ { ( y - \hat { y } _ { \tau } ) \tau , } & { L \leq y \leq R , } \\ { ( 1 - w _ { r } ) ( z _ { \infty } - \hat { y } _ { \tau } ) \tau + w _ { r } ( R - \hat { y } _ { \tau } ) \tau , } & { y > R } \end{array} \right. } \end{array}
$$

Takin<sub>g</sub> ex<sub>p</sub>ectation:

$$
\begin{array} { l } { \mathbb { E } _ { y } \big [ S _ { D P E } \big ( y ^ { * } , \hat { y } _ { \tau } , w _ { I } , w _ { r } , \tau \big ) \big ] } \\ { = \displaystyle \int _ { - \infty } ^ { L } \Big [ w _ { I } \rho _ { \tau } ( \hat { y } _ { \tau } , L ) + \big ( 1 - w _ { I } \big ) \rho _ { \tau } \big ( \hat { y } _ { \tau } , z _ { - \infty } \big ) \Big ] \hat { p } ( y ) d y } \\ { \displaystyle + \int _ { L } ^ { R } \rho _ { \tau } \big ( \hat { y } _ { \tau } , y \big ) \hat { p } ( y ) d y } \\ { + \displaystyle \int _ { R } ^ { \infty } \big [ w _ { r } \rho _ { \tau } ( \hat { y } _ { \tau } , R ) + \big ( 1 - w _ { r } \big ) \rho _ { \tau } \big ( \hat { y } _ { \tau } , z _ { \infty } \big ) \Big ] \hat { p } ( y ) d y } \\ { = \tau \mathbb { E } _ { L \leq y \leq R } [ y ] + \Big [ \big ( 1 - w _ { I } \big ) \tau _ { L } - \tau \Big ] \hat { y } _ { \tau } + C _ { 1 } \big ( \tau _ { L } , \tau _ { R } , w _ { I } , w _ { r } \big ) } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $C _ { 1 }$ <sub>co</sub>ll<sub>ec</sub>t<sub>s a</sub>ll t<sub>erms</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $\hat { y } _ { \tau }$ :

$$
\begin{array} { r } { C _ { 1 } = \tau _ { L } \left[ w _ { l } \tau L + ( 1 - w _ { l } ) ( \tau - 1 ) z _ { - \infty } \right] + ( 1 - \tau _ { R } ) \left[ w _ { r } \tau R + ( 1 - w _ { r } ) \tau z _ { \infty } \right] . } \end{array}
$$

S<sub>u</sub>bstit<sub>u</sub>tin<sub>g</sub> $\begin{array} { r } { w _ { l } = \frac { \tau _ { L } - \tau } { \tau _ { L } } } \end{array}$ <sub>,</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>o</sub>f $\hat { y } _ { \tau }$ <sub>van</sub>i<sub>s</sub>h<sub>es:</sub>

$$
\begin{array} { r } { \big ( 1 - w _ { l } \big ) \tau _ { L } - \tau = \tau - \tau _ { L } + \tau _ { L } - \tau = 0 . } \end{array}
$$

Th<sub>us</sub> $\mathbb { E } [ S _ { D P E } ( \hat { y } _ { \tau } ) ] = \tau \mathbb { E } _ { L \leq y \leq R } [ y ] + C _ { 1 }$ <sub>,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $\hat { y } _ { \tau }$ <sub>.</sub> Th<sub>e same</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> $y _ { \tau } , \mathrm { y }$ ieldin<sub>g</sub>:

$$
\mathbb { E } _ { y } [ S _ { D P E } ( y ^ { * } , \hat { y } _ { \tau } ) ] = \mathbb { E } _ { y } [ S _ { D P E } ( y ^ { * } , y _ { \tau } ) ] .
$$

Scenario 9: min $\{ \hat { y } _ { \tau } , y _ { \tau } \} > R$ . By symmetric argument with $w _ { r } =$ $\frac { \tau - \tau _ { R } } { 1 - \tau _ { R } }$ <sub>,</sub> th<sub>e</sub> $\hat { y } _ { \tau }$ <sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t b<sub>ecomes:</sub>

$$
\left[ \tau _ { R } - \tau + w _ { r } \big ( 1 - \tau _ { R } \big ) \right] \hat { y } _ { \tau } = \left[ \tau _ { R } - \tau + \tau - \tau _ { R } \right] \hat { y } _ { \tau } = 0 .
$$

H<sub>ence</sub> <sub>equa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> id<sub>en</sub>ti<sub>ca</sub>ll<sub>y.</sub>

7.1.2 Case II: One Prediction Inside [�, �], One Outside. This covers S<sub>ce</sub>n<sub>a</sub>ri<sub>os</sub> 2<sub>,</sub> 3<sub>,</sub> 4<sub>,</sub> 6<sub>,</sub> 7<sub>,</sub> 8<sub>.</sub> W<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il S<sub>ce</sub>n<sub>a</sub>ri<sub>o</sub> $2 ;$ <sub>o</sub>th<sub>ers</sub> f<sub>o</sub>ll<sub>ow</sub> b<sub>y</sub> s<sub>y</sub>mmetr<sub>y</sub> or combination.

Scenario $2 \colon \hat { y } _ { \tau } < L \leq y _ { \tau } \leq R$ . The DPE loss satisfies the lower b<sub>oun</sub>d<sub>:</sub>

$$
\begin{array}{c} \begin{array} { r l r } { S _ { D P E } ( y ^ { * } , \hat { y } _ { \tau } ) \geq } \\ { \left\{ { ( 1 - w _ { l } ) ( z _ { - \infty } - \hat { y } _ { \tau } ) ( \tau - 1 ) + w _ { l } ( L - \hat { y } _ { \tau } ) ( \tau - 1 ) } , } & { y < L , \right. } \\ { \left. ( y - \hat { y } _ { \tau } ) ( \tau - 1 ) , \right.} & { L \leq y < y _ { \tau } , } \\ { ( y - \hat { y } _ { \tau } ) \tau , } & { y _ { \tau } \leq y \leq R , } \\ { ( 1 - w _ { r } ) ( z _ { \infty } - \hat { y } _ { \tau } ) \tau + w _ { r } ( R - \hat { y } _ { \tau } ) \tau , } & { y > R } \end{array}   \end{array}
$$

where the ine<sub>q</sub>ualit<sub>y</sub> uses:

$$
\begin{array} { r l } & { \bullet ( y - \hat { y } _ { \tau } ) \tau \geq ( y - \hat { y } _ { \tau } ) ( \tau - 1 ) \mathrm { ~ f o r ~ } y \geq L > \hat { y } _ { \tau } ( \mathrm { s i n c e ~ } \tau > \tau - 1 } \\ & { \mathrm { ~ a n d ~ } y - \hat { y } _ { \tau } > 0 ) ; } \end{array}
$$

$$
\bullet w _ { l } ( L - \hat { y } _ { \tau } ) \tau \geq w _ { l } ( L - \hat { y } _ { \tau } ) ( \tau - 1 ) \mathrm { f o r } \hat { y } _ { \tau } < L .
$$

Takin<sub>g</sub> ex<sub>p</sub>ectations:

$$
\begin{array} { r l } & { \mathbb { E } [ S _ { D P E } ( \hat { y } _ { \tau } ) ] \geq \tau \mathbb { E } _ { L \leq y \leq R } [ y ] - \mathbb { E } _ { L \leq y < y _ { \tau } } [ y ] } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x } + \tau _ { L } \left[ w _ { l } ( \tau - 1 ) L + ( 1 - w _ { l } ) ( \tau - 1 ) z _ { - \infty } \right] } \\ & { \phantom { x x x x x x x x x x x x x x x x } + ( 1 - \tau _ { R } ) \left[ w _ { r } \tau R + \left( 1 - w _ { r } \right) \tau z _ { \infty } \right] . } \end{array}\tag{2}
$$

F<sub>o</sub>r th<sub>e</sub> tr<sub>ue</sub> <sub>p</sub>r<sub>e</sub>di<sub>c</sub>ti<sub>o</sub>n $y _ { \tau } \in [ L , R ]$ <sub>,</sub> the DPE e<sub>v</sub>al<sub>u</sub>ates exactl<sub>y</sub>:

�<sub>���</sub> (<sub>�</sub><sup>∗</sup>, <sub>��</sub> )

$$
= \left\{ \begin{array} { c c } { \big ( 1 - w _ { l } \big ) \big ( z _ { - \infty } - y _ { \tau } \big ) \big ( \tau - 1 \big ) + w _ { l } \big ( L - y _ { \tau } \big ) \big ( \tau - 1 \big ) , } & { \quad y < L , } \\ { \big ( y - y _ { \tau } \big ) \big ( \tau - 1 \big ) , } & { \quad L \leq y < y _ { \tau } , } \\ { \big ( y - y _ { \tau } \big ) \tau , } & { \quad y _ { \tau } \leq y \leq R , } \\ { \big ( 1 - w _ { r } \big ) \big ( z _ { \infty } - y _ { \tau } \big ) \tau + w _ { r } \big ( R - y _ { \tau } \big ) \tau , } & { \quad y > R } \end{array} \right.
$$

Com<sub>pu</sub>tin<sub>g</sub> its ex<sub>p</sub>ectation:

$$
\begin{array} { r l } & { \mathbb { E } [ S _ { D P E } ( y _ { \tau } ) ] = \tau \mathbb { E } _ { L \leq y \leq R } [ y ] - \mathbb { E } _ { L \leq y < y _ { \tau } } [ y ] } \\ & { \phantom { = } + \tau _ { L } \left[ w _ { l } ( \tau - 1 ) L + ( 1 - w _ { l } ) ( \tau - 1 ) z _ { - \infty } \right] } \\ & { \phantom { = } + ( 1 - \tau _ { R } ) \left[ w _ { r } \tau R + ( 1 - w _ { r } ) \tau z _ { \infty } \right] . } \end{array}\tag{3}
$$

Com<sub>p</sub>arin<sub>g</sub> (2) and (3), the ine<sub>q</sub>ualit<sub>y</sub> in (2) arises <sub>p</sub>recisel<sub>y</sub> because $\hat { y } _ { \tau } < L \leq y _ { \tau }$ f<sub>orces</sub> th<sub>e</sub> l<sub>oss</sub> t<sub>o use</sub> th<sub>e su</sub>b<sub>op</sub>ti<sub>ma</sub>l <sub>s</sub>l<sub>ope</sub> $( \tau - 1 )$ on $[ L , y _ { \tau } )$ <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>m</sub>i<sub>x.</sub> H<sub>ence:</sub>

$$
\mathbb { E } [ S _ { D P E } ( \hat { y } _ { \tau } ) ] \geq \mathbb { E } [ S _ { D P E } ( y _ { \tau } ) ] .
$$

Scenarios 3, 4, $6 , 7 , 8 .$ Th<sub>ese</sub> f<sub>o</sub>ll<sub>ow</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>pa</sub>tt<sub>erns:</sub>

• Scenario $3 ( \hat { y } _ { \tau } < L , y _ { \tau } > R )$ <sub>:</sub> C<sub>om</sub>bi<sub>nes</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-censore</sub>d lo<sub>g</sub>ic of Scenario 2 with ri<sub>g</sub>ht-censored lo<sub>g</sub>ic. The lower b<sub>oun</sub>d <sub>uses</sub> $( \tau - 1 )$ <sub>on</sub> b<sub>o</sub>th t<sub>a</sub>il<sub>s, w</sub>hil<sub>e</sub> $y _ { \tau }$ uses o<sub>p</sub>timal <sub>�</sub> on [�, �] and $( \tau - 1 )$ <sub>on</sub>l<sub>y</sub> f<sub>or</sub> $y > y _ { \tau }$

• Scenario $4 ( L \leq \hat { y } _ { \tau } \leq R , y _ { \tau } < L ) \colon \mathrm { S } ;$ <sub>y</sub>mm<sub>e</sub>tri<sub>c</sub> t<sub>o</sub> S<sub>ce</sub>n<sub>a</sub>ri<sub>o</sub> 2 b<sub>y</sub> exchan<sub>g</sub>in<sub>g</sub> left/ri<sub>g</sub>ht and $\tau  ( \tau - 1 )$

• Scenario 6 $( L \leq \hat { y } _ { \tau } \leq R , y _ { \tau } > R ) \colon$ : S<sub>y</sub>mmetric to Scenario 4 <sub>on</sub> th<sub>e r</sub>i<sub>g</sub>ht t<sub>a</sub>il<sub>.</sub>

• Scenario $7 \left( \hat { y } _ { \tau } > R , y _ { \tau } < L \right)$ : S<sub>y</sub>mm<sub>e</sub>tri<sub>c</sub> t<sub>o</sub> S<sub>ce</sub>n<sub>a</sub>ri<sub>o</sub> 3<sub>.</sub>

• Scenario 8 $( \hat { y } _ { \tau } > R , L \leq y _ { \tau } \leq R ) \colon$ : S<sub>y</sub>mmetric to Scenario 2 <sub>on</sub> th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht t<sub>a</sub>il<sub>.</sub>

I<sub>n</sub> <sub>eac</sub>h <sub>case,</sub> th<sub>e</sub> <sub>m</sub>i<sub>spre</sub>di<sub>c</sub>t<sub>e</sub>d $\hat { y } _ { \tau }$ li<sub>es</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> $[ L , R ]$ <sub>an</sub>d i<sub>ncurs</sub> <sub>a</sub> <sub>su</sub>b<sub>op</sub>ti<sub>ma</sub>l li<sub>near com</sub>bi<sub>na</sub>ti<sub>on o</sub>f <sub>s</sub>l<sub>opes, w</sub>hil<sub>e</sub> $y _ { \tau } \in [ L , R ]$ <sub>ac</sub>hi<sub>eves</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>p</sub>i<sub>n</sub>b<sub>a</sub>ll l<sub>oss</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure.</sub> □

7.1.3 Case III: Both Predictions Inside $[ L , R ]$ . This covers Scenarios $5 \colon L \leq \hat { y } _ { \tau } \leq R , L < y _ { \tau } < R .$

Scenario 5. Without loss of generality, assume $\hat { y } _ { \tau } < y _ { \tau }$ (the case $\hat { y } _ { \tau } > y _ { \tau }$ is s mmetric). The DPE loss satisfies:

$$
\begin{array} { r l } & { S _ { D P E } ( y ^ { * } , \hat { y } _ { \tau } ) } \\ & { \geq \left\{ \begin{array} { c c } { ( 1 - w _ { l } ) ( z _ { - \infty } - \hat { y } _ { \tau } ) ( \tau - 1 ) + w _ { l } ( L - \hat { y } _ { \tau } ) ( \tau - 1 ) , } & { y < L , } \\ { ( y - \hat { y } _ { \tau } ) ( \tau - 1 ) , } & { L \leq y < y _ { \tau } , } \\ { ( y - \hat { y } _ { \tau } ) \tau , } & { y _ { \tau } \leq y \leq R , } \\ { ( 1 - w _ { r } ) ( z _ { \infty } - \hat { y } _ { \tau } ) \tau + w _ { r } ( R - \hat { y } _ { \tau } ) \tau , } & { y > R } \end{array} \right. } \end{array}
$$

Th<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> b<sub>ecause:</sub>

• For $y \in \left[ { \hat { y } } _ { \tau } , y _ { \tau } \right) : ( y - { \hat { y } } _ { \tau } ) \tau \geq \left( y - { \hat { y } } _ { \tau } \right) \left( \tau - 1 \right)$ <sub>s</sub>in<sub>ce</sub> $y - \hat { y } _ { \tau } > 0 ;$

• For $y \in [ L , \hat { y } _ { \tau } ) \colon ( y - \hat { y } _ { \tau } ) ( \tau - 1 ) > ( y - \hat { y } _ { \tau } ) \cdot$ <sub>� s</sub>in<sub>ce</sub> $y - \hat { y } _ { \tau } < 0$ <sub>an</sub>d $\tau - 1 < \tau ,$ b<sub>u</sub>t th<sub>e</sub> <sub>overa</sub>ll <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>preserves</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d d<sub>ue</sub> t<sub>o</sub> th<sub>e</sub> <sub>we</sub>i<sub>g</sub>hti<sub>ng.</sub>

Takin<sub>g</sub> ex<sub>p</sub>ectations and usin<sub>g</sub> $\begin{array} { r } { w _ { l } = \frac { \tau _ { L } - \tau } { \tau _ { L } } , w _ { r } = \frac { \tau - \tau _ { R } } { 1 - \tau _ { R } } } \end{array}$

$$
\begin{array} { r l } & { \mathbb { E } \big [ S _ { D P E } ( \hat { y } _ { \tau } ) \big ] \geq \mathbb { E } _ { L < y < R } [ y ] - \mathbb { E } _ { L < y < y _ { \tau } } \big [ y \big ] } \\ & { \phantom { = } + \tau _ { L } \Big [ \big ( 1 - w _ { l } \big ) \big ( \tau - 1 \big ) z _ { - \infty } + w _ { l } \big ( \tau - 1 \big ) L \Big ] } \\ & { \phantom { = } + \big ( 1 - \tau _ { R } \big ) \Big [ \big ( 1 - w _ { r } \big ) \tau z _ { \infty } + w _ { r } \tau R \Big ] . } \end{array}\tag{4}
$$

For the tr<sub>u</sub>e <sub>p</sub>rediction <sub>��</sub>:

$$
\begin{array} { r l } & { S _ { D P E } ( y ^ { * } , y _ { \tau } ) } \\ & { = \left\{ \begin{array} { c c } { ( 1 - w _ { l } ) ( z _ { - \infty } - y _ { \tau } ) ( \tau - 1 ) + w _ { l } ( L - y _ { \tau } ) ( \tau - 1 ) , } & { y < L , } \\ { ( y - y _ { \tau } ) ( \tau - 1 ) , } & { L \leq y < y _ { \tau } , } \\ { ( y - y _ { \tau } ) \tau , } & { y _ { \tau } \leq y \leq R , } \\ { ( 1 - w _ { r } ) ( z _ { \infty } - y _ { \tau } ) \tau + w _ { r } ( R - y _ { \tau } ) \tau , } & { y > R } \end{array} \right. } \end{array}
$$

Its ex<sub>p</sub>ectation evaluates to exactl<sub>y</sub> the ri<sub>g</sub>ht-hand side of (4):

$$
\begin{array} { r l } & { \mathbb { E } \big [ S _ { D P E } ( y _ { \tau } ) \big ] = \mathbb { E } _ { L < y < R } [ y ] - \mathbb { E } _ { L < y < y _ { \tau } } [ y ] } \\ & { \phantom { = } + \tau _ { L } \left[ ( 1 - w _ { l } ) ( \tau - 1 ) z _ { - \infty } + w _ { l } ( \tau - 1 ) L \right] } \\ & { \phantom { = } + ( 1 - \tau _ { R } ) \left[ ( 1 - w _ { r } ) \tau z _ { \infty } + w _ { r } \tau R \right] . } \end{array}
$$

Th<sub>ere</sub>f<sub>ore:</sub>

$$
\mathbb { E } [ S _ { D P E } ( \hat { y } _ { \tau } ) ] \geq \mathbb { E } [ S _ { D P E } ( y _ { \tau } ) ] .
$$

Wh<sub>en</sub> $\hat { y } _ { \tau } = y _ { \tau } ,$ <sub>,</sub> all ine<sub>q</sub>ualities become e<sub>q</sub>ualities<sub>,</sub> confirmin<sub>g</sub> p<sup>ro</sup>p<sup>erness.</sup> □

## 7.2 Summary

A<sub>c</sub>r<sub>oss a</sub>ll <sub>geo</sub>m<sub>e</sub>tri<sub>c co</sub>nfi<sub>gu</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>s</sub>:

$$
\mathbb { E } _ { y } [ S _ { D P E } ( y ^ { * } , \hat { y } _ { \tau } , w _ { l } , w _ { r } , \tau ) ] \ge \mathbb { E } _ { y } [ S _ { D P E } ( y ^ { * } , y _ { \tau } , w _ { l } , w _ { r } , \tau ) ]\tag{5}
$$

with e<sub>q</sub>ualit<sub>y</sub> if and onl<sub>y</sub> $\mathrm { i f } \hat { y } _ { \tau } = y _ { \tau }$ <sub>.</sub> Th<sub>us</sub> DPE i<sub>s</sub> <sub>a</sub> <sub>p</sub>r<sub>ope</sub>r <sub>sco</sub>rin<sub>g</sub> <sub>ru</sub>l<sub>e.</sub> □

## 7.3 Proof of Theorem 2

Setup. Let � be the true CDF of �, $\tau _ { y } = F ( y ) \sim U [ 0 , 1 ]$ <sub>,</sub> i<sub>n</sub>t<sub>erva</sub>l $I = [ \tau _ { k } , \tau _ { k + 1 } )$ . Under inde<sub>p</sub>endent censorin<sub>g</sub> $y \perp ( L , R )$ <sub>, we</sub> h<sub>ave</sub> $P ( y > y _ { u } ) = 1 - u$ <sub>an</sub>d $P ( y < y _ { u } ) = u$

Step 1. Write the expectation as five terms:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \frac { N _ { k } ^ { D } } { N } \right] = P [ \left( \tau _ { y } \in I \right) \wedge \left( L < y < R \right) ] } \\ & { + \frac { \tau _ { k + 1 } - \tau _ { R } } { 1 - \tau _ { R } } P [ \left( \tau _ { R } \in I \right) \wedge \left( y > R \right) ] + \frac { \tau _ { k + 1 } - \tau _ { k } } { 1 - \tau _ { R } } P [ \left( \tau _ { R } < \tau _ { k } \right) \wedge \left( y > R \right) ] } \\ & { + \frac { \tau _ { L } - \tau _ { k } } { \tau _ { L } } P [ \left( \tau _ { L } \in I \right) \wedge \left( y < L \right) ] + \frac { \tau _ { k + 1 } - \tau _ { k } } { \tau _ { L } } P [ \left( \tau _ { L } > \tau _ { k + 1 } \right) \wedge \left( y < L \right) ] } \end{array}
$$

Step 2. Decompose the interval probabilities using $\tau _ { y } \sim U [ 0 , 1 ]$

$$
\begin{array} { r l } & { P [ ( \tau _ { R } \in I ) \wedge ( y > R ) ] = ( 1 - \tau _ { k + 1 } ) P ( \tau _ { R } \in I ) } \\ & { \qquad + P [ ( ( \tau _ { R } \wedge \tau _ { y } ) \in I ) \wedge ( y > R ) ] } \\ & { P [ ( \tau _ { L } \in I ) \wedge ( y < L ) ] = \tau _ { k } P ( \tau _ { L } \in I ) } \\ & { \qquad + P [ ( ( \tau _ { L } \wedge \tau _ { y } ) \in I ) \wedge ( y < L ) ] } \end{array}
$$

Step 3. Combine right-censored terms (a+b):

$$
\begin{array} { r l } & { \frac { \tau _ { k + 1 } - 1 } { 1 - \tau _ { R } } P [ ( \tau _ { R } \in I ) \wedge ( y > R ) ] + \frac { \tau _ { k + 1 } - \tau _ { k } } { 1 - \tau _ { R } } P [ ( \tau _ { R } < \tau _ { k } ) \wedge ( y > R ) ] } \\ & { = \tau _ { k + 1 } P ( \tau _ { R } < \tau _ { k + 1 } ) - P ( \tau _ { R } \in I ) - \tau _ { k } P ( \tau _ { R } < \tau _ { k } ) } \\ & { = ( \tau _ { k + 1 } - \tau _ { k } ) P ( \tau _ { R } < \tau _ { k } ) + ( \tau _ { k + 1 } - 1 ) P ( \tau _ { R } \in I ) } \end{array}
$$

Step 4. Combine left-censored terms (c+d):

$$
\begin{array} { r l } & { \frac { \tau _ { k + 1 } - \tau _ { k } } { \tau _ { L } } P [ ( \tau _ { L } > \tau _ { k + 1 } ) \wedge ( y < L ) ] - \frac { \tau _ { k } } { \tau _ { L } } P [ ( \tau _ { L } \in I ) \wedge ( y < L ) ] } \\ & { = \tau _ { k + 1 } P ( \tau _ { L } > \tau _ { k + 1 } ) - \tau _ { k } P ( \tau _ { L } > \tau _ { k } ) } \\ & { = ( \tau _ { k + 1 } - \tau _ { k } ) P ( \tau _ { L } > \tau _ { k + 1 } ) - \tau _ { k } P ( \tau _ { L } \in I ) } \end{array}
$$

Step 5. Combine overlapping terms with uncensored:

$$
\begin{array} { r l } & { P \big [ ( ( \tau _ { R } \wedge \tau _ { y } ) \in I ) \wedge ( y > R ) \big ] + P \big [ ( ( \tau _ { L } \wedge \tau _ { y } ) \in I ) \wedge ( y < L ) \big ] } \\ & { \quad + P \big [ ( \tau _ { y } \in I ) \wedge ( L < y < R ) \big ] } \\ & { = P \big [ ( \tau _ { y } \in I ) \wedge ( R > y _ { \tau _ { k } } ) \wedge ( L < y _ { \tau _ { k + 1 } } ) \big ] } \\ & { = ( \tau _ { k + 1 } - \tau _ { k } ) P \big [ ( R > y _ { \tau _ { k } } ) \wedge ( L < y _ { \tau _ { k + 1 } } ) \big ] } \end{array}
$$

Step 6. Assemble all pieces:

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \frac { N _ { k } ^ { D } } { N } \right] = ( 1 - \tau _ { k + 1 } ) P ( \tau _ { R } \in I ) + \tau _ { k } P ( \tau _ { L } \in I ) } \\ { \displaystyle \qquad + ( \tau _ { k + 1 } - \tau _ { k } ) P [ ( R > y _ { \tau _ { k } } ) \wedge ( L < y _ { \tau _ { k + 1 } } ) ] } \\ { \displaystyle \qquad + ( \tau _ { k + 1 } - \tau _ { k } ) P ( \tau _ { R } < \tau _ { k } ) + ( \tau _ { k + 1 } - 1 ) P ( \tau _ { R } \in I ) } \\ { \displaystyle \qquad + ( \tau _ { k + 1 } - \tau _ { k } ) P ( \tau _ { L } > \tau _ { k + 1 } ) - \tau _ { k } P ( \tau _ { L } \in I ) } \end{array}
$$

Step 7. Cancel boundary terms: $\left( 1 - \tau _ { k + 1 } \right) + \left( \tau _ { k + 1 } - 1 \right) = 0$ f<sub>or</sub> $P ( \tau _ { R } \in I )$ <sub>,</sub> <sub>an</sub>d $\tau _ { k } - \tau _ { k } = 0$ f<sub>or</sub> $P ( \tau _ { L } \in I )$

Step 8. Final simplification:

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \frac { N _ { k } ^ { D } } { N } \right] = ( \tau _ { k + 1 } - \tau _ { k } ) \big \{ P [ ( R > y _ { \tau _ { k } } ) \land ( L < y _ { \tau _ { k + 1 } } ) ] } \\ { \quad \quad \quad + P ( \tau _ { R } < \tau _ { k } ) + P ( \tau _ { L } > \tau _ { k + 1 } ) \big \} } \\ { = ( \tau _ { k + 1 } - \tau _ { k } ) \cdot 1 = \tau _ { k + 1 } - \tau _ { k } } \end{array}
$$

The three events <sub>p</sub>artition the sam<sub>p</sub>le s<sub>p</sub>ace: (i) <sub>�</sub> observable in �, (ii) ri<sub>g</sub>ht-censored into �, (iii) left-censored into �.

$$
\therefore \mathbb { E } [ N _ { k } ^ { D } / N ] = F ( y _ { \tau _ { k + 1 } } ) - F ( y _ { \tau _ { k } } )
$$

## 7.4 Detailed Derivation of Inventory Optimization

Th<sub>e s</sub>t<sub>oc</sub>k<sub>ou</sub>t l<sub>oss can</sub> b<sub>e</sub> d<sub>er</sub>i<sub>ve</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows,</sub>

$$
\begin{array} { l } { \displaystyle L _ { 1 } ( q ) = \int _ { t } ^ { \infty } c _ { 1 } ( y - t ) p ( y | x ) d y } \\ { \displaystyle = c _ { 1 } \int _ { t } ^ { \infty } y p ( y | x ) d y - t \big [ 1 - F ( t | x ) \big ] } \\ { \displaystyle = c _ { 1 } \int _ { F ( t | x ) } ^ { 1 } F ^ { - 1 } ( q | x ) d q - t \big [ 1 - F ( t | x ) \big ] } \\ { \displaystyle = c _ { 1 } \sum _ { q = q _ { t } } ^ { 1 } \hat { F } ^ { - 1 } ( q | x ) \cdot \Delta q - c _ { 1 } \big \{ \hat { F } ^ { - 1 } ( q _ { t } | x ) \big [ 1 - q _ { t } \big ] \big \} } \end{array}
$$

Wh<sub>ere</sub> ${ \hat { F } } ^ { - 1 }$ i<sub>s</sub> th<sub>e es</sub>tim<sub>a</sub>t<sub>e o</sub>f th<sub>e</sub> in<sub>ve</sub>r<sub>se</sub> CDF<sub>,</sub> $q _ { t } \ = \ \hat { F } ( t | x )$ <sub>an</sub>d $\Delta q = 0 . 0 1$ . For the last e<sub>q</sub>uation<sub>,</sub> we levera<sub>g</sub>e the Riemann sum to di<sub>scre</sub>ti<sub>ze</sub> th<sub>e</sub> i<sub>n</sub>t<sub>egra</sub>l<sub>.</sub> Si<sub>m</sub>il<sub>ar</sub>l<sub>y,</sub> th<sub>e overs</sub>t<sub>oc</sub>k l<sub>oss can</sub> b<sub>e</sub> d<sub>er</sub>i<sub>ve</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows,</sub>

$$
\begin{array} { l } { \displaystyle { L _ { 2 } ( q ) = \int _ { 0 } ^ { t } c _ { 2 } ( t - y ) \hat { p } ( y | x ) d y } } \\ { \displaystyle { \quad = c _ { 2 } \cdot t \cdot F ( t ) - c _ { 2 } \int _ { 0 } ^ { t } y \hat { p } ( y | x ) d y } } \\ { \displaystyle { \quad = c _ { 2 } \big [ q _ { t } \cdot \hat { F } ^ { - 1 } ( q _ { t } | x ) - \sum _ { q = 0 } ^ { q _ { t } } \hat { F } ^ { - 1 } ( q | x ) \cdot \Delta q \big ] } } \end{array}
$$

## GenAI Usage Disclosure

The authors declare that a <sub>g</sub>eneral-<sub>p</sub>ur<sub>p</sub>ose LLM (Kimi) was used <sub>exc</sub>l<sub>us</sub>i<sub>ve</sub>l<sub>y</sub> f<sub>or</sub> l<sub>anguage re</sub>fi<sub>nemen</sub>t <sub>o</sub>f th<sub>e au</sub>th<sub>or-</sub>d<sub>ra</sub>ft<sub>e</sub>d t<sub>ex</sub>t<sub>.</sub> N<sub>o</sub> AI tools were utilized in the research <sub>p</sub>rocess<sub>,</sub> includin<sub>g</sub> <sub>q</sub>uestion formulation<sub>,</sub> codin<sub>g,</sub> ex<sub>p</sub>erimentation<sub>,</sub> or data anal<sub>y</sub>sis<sub>,</sub> and thus th<sub>e</sub> AI <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d <sub>no</sub> i<sub>n</sub>t<sub>e</sub>ll<sub>ec</sub>t<sub>ua</sub>l <sub>con</sub>t<sub>en</sub>t t<sub>o</sub> th<sub>e wor</sub>k<sub>.</sub>

## References

[1] Anand Avati, Ton<sub>y</sub> Duan, Sharon Zhou, Kenneth Jun<sub>g</sub>, Ni<sub>g</sub>am H Shah, and Andrew Y N<sub>g</sub>. 2020. Countdown re<sub>g</sub>ression: shar<sub>p</sub> and calibrated survival <sub>p</sub>redictions. In Uncertainty in Artificial Intelligence. PMLR 145–155.

[2] Gah-Yi Ban and C<sub>y</sub>nthia Rudin. 2019. The bi<sub>g</sub> data newsvendor: Practical insi<sub>g</sub>hts from machine learning. Operations Research 67, 1 (2019), 90–108.

[3] Jonathan Berrisch and Florian Ziel. 2023. CRPS learning. Journal of Econometrics 237, 2 (2023), 105221.

[4] Anna-Lena Beutel and Stefan Minner. 2012. Safet<sub>y</sub> stock <sub>p</sub>lannin<sub>g</sub> under causal demand forecasting. International Journal ofProduction Economics 140, 2 (2012), 637–645.

[5] Axel Brando, Barcelona Su<sub>p</sub>ercom<sub>p</sub>utin<sub>g</sub> Center, Jose Rodri<sub>g</sub>uez-Serrano, Jordi Vitrià<sub>,</sub> et al. 2022. Dee<sub>p</sub> non-crossin<sub>g q</sub>uantiles throu<sub>g</sub>h the <sub>p</sub>artial derivative. In International Conference on Artificial Intelligence and Statistics. PMLR, 7902–7914.

[6] Alex J. Cannon. 2018. Non-crossin<sub>g</sub> nonlinear re<sub>g</sub>ression <sub>q</sub>uantiles b<sub>y</sub> monotone com<sub>p</sub>osite <sub>q</sub>uantile re<sub>g</sub>ression neural network<sub>,</sub> with a<sub>pp</sub>lication to rainfall extremes. Stochastic Environmental Research and Risk Assessment 32(11) (2018), 3207–3225. doi:10.1007/s00477-018-1573-6

[7] Si-An Chen, Chun-Lian<sub>g</sub> Li, Nate Yoder, Sercan O Arik, and Tomas Pfister. 2023. Tsmixer: An all-mlp architecture for time series forecasting. arXiv preprint arXiv:2303.06053 (2023).

[8] Youn<sub>g</sub>seo<sub>g</sub> Chun<sub>g</sub>, Willie Neiswan<sub>g</sub>er, Ian Char, and Jef Schneider. 2021. Be<sub>y</sub>ond pinball loss: Quantile methods for calibrated uncertainty quantification. Advances in Neural Information Processing Systems 34 (2021), 10971–10984.

[9] Pen<sub>g</sub> Cui, Wenbo Hu, and Jun Zhu. 2020. Calibrated reliable re<sub>g</sub>ression usin<sub>g</sub> maximum mean discrepancy. Advances in Neural Information Processing Systems 33 (2020), 17164–17175.

[10] Shai Feldman, Ste<sub>p</sub>hen Bates, and Yaniv Romano. 2021. Im<sub>p</sub>rovin<sub>g</sub> conditional coverage via orthogonal quantile regression. Advances in neural information processing systems 34 (2021), 2060–2071.

[11] Jan Gasthaus, Konstantinos Benidis, Yu<sub>y</sub>an<sub>g</sub> Wan<sub>g</sub>, S<sub>y</sub>ama Sundar Ran<sub>g</sub>a<sub>p</sub>uram, David Salinas, Valentin Flunkert, and Tim Januschowski. 2019. Probabilistic forecasting with spline quantile function RNNs. In The 22nd international conference on artificial intelligence and statistics. PMLR, 1901–1910.

[12] Tilmann Gneitin<sub>g</sub> and Adrian E Rafter<sub>y</sub>. 2007. Strictl<sub>y</sub> <sub>p</sub>ro<sub>p</sub>er scorin<sub>g</sub> rules, prediction, and estimation. Journal of the American statistical Association 102, 477 (2007), 359–378.

[13] Mark Goldstein, Xintian Han, Aahlad Puli, Adler Perotte, and Rajesh Ran<sub>g</sub>anath. 2020. X-cal: Explicit calibration for survival analysis. Advances in neural information processing systems 33 (2020), 18296–18307.

[14] Z Govindarajulu. 1992. Rank correlation methods.

[15] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q Weinber<sub>g</sub>er. 2017. On calibration of modern neural networks. In International conference on machine learning. PMLR, 1321–1330.

[16] Humza Haider, Bret Hoehn, Sarah Davis, and Russell Greiner. 2020. Efective ways to build and evaluate individual survival distributions. Journal ofMachine Learning Research 21, 85 (2020), 1–63.

[17] Frank E Harrell, Robert M Calif, David B Pr<sub>y</sub>or, Kerr<sub>y</sub> L Lee, and Robert A Rosati. 1982. Evaluating the yield of medical tests. Jama 247, 18 (1982), 2543–2546.

[18] Shuan<sub>g</sub> Ji, Limin Pen<sub>g</sub>, Yu Chen<sub>g</sub>, and HuiChuan Lai. 2012. Quantile re<sub>g</sub>ression for doubl censored data. Biometrics 68, 1 (2012), 101–112.

[19] Jared L Katzman, Uri Shaham, Alexander Clonin<sub>g</sub>er, Jonathan Bates, Tin<sub>g</sub>tin<sub>g</sub> Jian<sub>g</sub>, and Yuval Klu<sub>g</sub>er. 2018. Dee<sub>p</sub>Surv: <sub>p</sub>ersonalized treatment recommender system using a Cox proportional hazards deep neural network. BMC medical research methodology 18 (2018), 1–12.

[20] Roger Koenker and Gilbert Bassett Jr. 1978. Regression quantiles. Econometrica: journal of the Econometric Society (1978), 33–50.

[21] RetsefLevi, Geor<sub>g</sub>ia Perakis, and Joline Uichanco. 2015. The data-driven newsvendor problem: New bounds and insights. Operations Research 63, 6 (2015), 1294– 1306.

[22] Br<sub>y</sub>an Lim, Sercan Ö Arık, Nicolas Loef, and Tomas Pfister. 2021. Tem<sub>p</sub>oral fusion transformers for interpretable multi-horizon time series forecasting. International Journal ofForecasting 37, 4 (2021), 1748–1764.

[23] Guixian Lin, Xumin<sub>g</sub> He, and Ste<sub>p</sub>hen Portno<sub>y</sub>. 2012. Quantile re<sub>g</sub>ression with doubly censored data. Computational Statistics & Data Analysis 56, 4 (2012), 797–812.

[24] Yu<sub>q</sub>i Nie, Nam H N<sub>g</sub>u<sub>y</sub>en, Phanwadee Sinthon<sub>g</sub>, and Ja<sub>y</sub>ant Kala<sub>g</sub>nanam. 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730 (2022).

[25] Youn<sub>g</sub>suk Park, Danielle Maddix, François-Xavier Aubet, Kelvin Kan, Jan Gasthaus<sub>,</sub> and Yu<sub>y</sub>an<sub>g</sub> Wan<sub>g</sub>. 2022. Learnin<sub>g q</sub>uantile functions without <sub>q</sub>uantile crossing for distribution-free time series forecasting. In International Conference on Artificial Intelligence and Statistics. PMLR, 8127–8150.

[26] Tim Pearce, Jon<sub>g</sub>-H<sub>y</sub>eon Jeon<sub>g</sub>, Jun Zhu, et al. 2022. Censored <sub>q</sub>uantile re<sub>g</sub>ression neural networks for distribution-free survival analysis. Advances in Neural Information Processing Systems 35 (2022), 7450–7461.

[27] Limin Peng. 2021. Quantile regression for survival data. Annual review ofstatistics and its application 8 (2021), 413–437.

[28] Fotios Petro<sub>p</sub>oulos, Daniele A<sub>p</sub>iletti, Vassilios Assimako<sub>p</sub>oulos, Mohamed Zied Babai, Devon K Barrow, Souhaib Ben Taieb, Christo<sub>p</sub>h Ber<sub>g</sub>meir, Ricardo J Bessa, Jakub Bijak, John E Boylan, et al. 2022. Forecasting: theory and practice. International Journal ofForecasting 38, 3 (2022), 705–871.

[29] Meng Qi, Yuanyuan Shi, Yongzhi Qi, Chenxin Ma, Rong Yuan, Di Wu, and Zuo-Jun Shen. 2023. A <sub>p</sub>ractical end-to-end inventor<sub>y</sub> mana<sub>g</sub>ement model with dee<sub>p</sub> learning. Management Science 69, 2 (2023), 759–773.

[30] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. 2020. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International journal offorecasting 36, 3 (2020), 1181–1191.

[31] Charles S<sub>p</sub>earman. 1961. The <sub>p</sub>roof and measurement of association between two thin<sub>g</sub>s. (1961).

[32] Souhaib Ben Taieb. 2022. Learnin<sub>g</sub> <sub>q</sub>uantile functions for tem<sub>p</sub>oral <sub>p</sub>oint <sub>p</sub>rocesses with recurrent neural splines. In International Conference on Artificial Intelligence and Statistics. PMLR, 3219–3241.

[33] Kevin Tran, Willie Neiswan<sub>g</sub>er, Junwoon<sub>g</sub> Yoon, Qin<sub>gy</sub>an<sub>g</sub> Zhan<sub>g</sub>, Eric Xin<sub>g</sub>, and Zachar<sub>y</sub> W Ulissi. 2020. Methods for com<sub>p</sub>arin<sub>g</sub> uncertaint<sub>y q</sub>uantifications

for material property predictions. Machine Learning: Science and Technology 1, 2 (2020), 025006.

[34] Huixia Jud<sub>y</sub> Wan<sub>g</sub> and Lan Wan<sub>g</sub>. 2009. Locall<sub>y</sub> wei<sub>g</sub>hted censored <sub>q</sub>uantile regression. J. Amer. Statist. Assoc. 104, 487 (2009), 1117–1128.

[35] Pin<sub>g</sub> Wan<sub>g</sub>, Yan Li, and Chandan K Redd<sub>y</sub>. 2019. Machine learnin<sub>g</sub> for survival analysis: A survey. ACM Computing Surveys (CSUR) 51, 6 (2019), 1–36.

[36] Ruofen<sub>g</sub> Wen, Kari Torkkola, Balakrishnan Nara<sub>y</sub>anaswam<sub>y</sub>, and Dhruv Madeka. 2017. A multi-horizon quantile recurrent forecaster. arXiv preprint arXiv:1711.11053 (2017).

[37] Frank Wilcoxon. 1992. Individual comparisons by ranking methods. In Breakthroughs in statistics: Methodology and distribution. Springer, 196–202.

[38] C. F. Jef Wu. 1983. On the Convergence Properties of the EM Algorithm. The Annals ofStatistics 11, 1 (1983), 95–103.

[39] Hiroki Yanagisawa. 2023. Proper scoring rules for survival analysis. In International Conference on Machine Learning. PMLR, 39165–39182.

[40] Qixian Zhong, Jonas W Mueller, and Jane-Ling Wang. 2021. Deep extended hazard models for survival analysis. Advances in Neural Information Processing Systems 34 (2021), 15111–15124.