# Dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n-Fr<sub>ee</sub> Pri<sub>va</sub>t<sub>e</sub> M<sub>ea</sub>n E<sub>s</sub>tim<sub>a</sub>ti<sub>o</sub>n i<sub>s</sub> Im<sub>poss</sub>ibl<sub>e</sub> With<sub>ou</sub>t th<sub>e</sub> C<sub>ova</sub>ri<sub>a</sub>n<sub>ce</sub>

R<sub>enau</sub>d G<sub>auc</sub>h<sub>er</sub>

E<sub>p</sub>hr<sub>a</sub>im Lind<sub>e</sub>r<sup>†</sup>

Ad<sub>am</sub> S<sub>m</sub>ith<sup>‡</sup>

Jonathan Ullman<sup>§</sup>

L<sub>y</sub>di<sub>a</sub> Z<sub>a</sub>k<sub>y</sub>nthin<sub>ou</sub><sup>¶</sup>

N<sub>ovem</sub>b<sub>er</sub> 5<sub>,</sub> 2026

## Abstract

We stud<sub>y</sub> the limits of diferentiall<sub>y</sub> <sub>p</sub>rivate Gaussian mean estimation when the covariance matrix is unknown. Prior work shows that when the covariance Σ is known, the o<sub>p</sub>timal sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> can de<sub>p</sub>end onl<sub>y</sub> on its s<sub>p</sub>ectrum<sub>,</sub> rather than on the ambient dimension �. We <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>new</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> G<sub>auss</sub>i<sub>an mean es</sub>ti<sub>ma</sub>ti<sub>on w</sub>ith E<sub>uc</sub>lid<sub>ean error w</sub>hi<sub>c</sub>h <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>p</sub>ri<sub>va</sub>t<sub>e es</sub>tim<sub>a</sub>t<sub>o</sub>r<sub>s ca</sub>nn<sub>o</sub>t <sub>ac</sub>hi<sub>eve s</sub>imil<sub>a</sub>r <sub>gua</sub>r<sub>a</sub>nt<sub>ees w</sub>h<sub>e</sub>n th<sub>e cova</sub>ri<sub>a</sub>n<sub>ce</sub> i<sub>s u</sub>nkn<sub>ow</sub>n<sub>.</sub> Fir<sub>s</sub>t<sub>,</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>s</sub>i<sub>ze</sub> <sub>�</sub> i<sub>s</sub> <sub>sma</sub>ll<sub>er</sub> th<sub>an</sub> th<sub>e</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> di<sub>mens</sub>i<sub>on</sub> t<sub>race</sub> $( \Sigma ) / \| \Sigma \| _ { \mathrm { o p } } ,$ th<sub>e op</sub>ti<sub>ma</sub>l <sub>error</sub> <sub>w</sub>h<sub>en</sub> th<sub>e covar</sub>i<sub>ance</sub> i<sub>s un</sub>k<sub>nown can</sub> b<sub>e</sub> l<sub>arger</sub> th<sub>an</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>error w</sub>h<sub>en</sub> it i<sub>s</sub> k<sub>nown</sub> b<sub>y a</sub> f<sub>ac</sub>t<sub>or</sub> <sub>o</sub>f ${ \sqrt { d / n } } .$ <sub>.</sub> S<sub>econ</sub>d<sub>,</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>s</sub>i<sub>ze</sub> i<sub>s</sub> l<sub>arger</sub> th<sub>an</sub> th<sub>e</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> di<sub>mens</sub>i<sub>on,</sub> <sub>even</sub> f<sub>or</sub> di<sub>agona</sub>l <sub>covar</sub>i<sub>ance</sub> <sub>ma</sub>t<sub>r</sub>i<sub>ces</sub> <sub>w</sub>ith <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> di<sub>mens</sub>i<sub>on</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> l<sub>og</sub> �<sub>,</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>error</sub> <sub>w</sub>h<sub>en</sub> th<sub>e covar</sub>i<sub>ance</sub> i<sub>s un</sub>k<sub>nown can</sub> b<sub>e</sub> l<sub>arger</sub> th<sub>an</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>error w</sub>h<sub>en</sub> it i<sub>s</sub> k<sub>nown</sub> b<sub>y a</sub> f<sub>ac</sub>t<sub>or</sub> <sub>o</sub>f $\sqrt { d } / n ^ { 3 / 2 }$ . To<sub>g</sub>ether<sub>,</sub> these results show that in the unknown covariance settin<sub>g,</sub> ex<sub>p</sub>loitin<sub>g</sub> anisotro<sub>p</sub>ic structure under diferential <sub>p</sub>rivac<sub>y</sub> can re<sub>q</sub>uire a dimension-de<sub>p</sub>endent error or sam<sub>p</sub><sup>l</sup>e com<sub>p</sub><sup>l</sup>ex<sup>i</sup>t<sub>y</sub>.

## 1 Introduction

Statistical inference and machine learnin<sub>g</sub> al<sub>g</sub>orithms are desi<sub>g</sub>ned to extract <sub>p</sub>o<sub>p</sub>ulation-level insi<sub>g</sub>hts f<sub>rom</sub> d<sub>a</sub>t<sub>a,</sub> b<sub>u</sub>t i<sub>n</sub> d<sub>o</sub>i<sub>ng so,</sub> th<sub>ey rou</sub>ti<sub>ne</sub>l<sub>y revea</sub>l i<sub>n</sub>f<sub>orma</sub>ti<sub>on a</sub>b<sub>ou</sub>t th<sub>e</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l<sub>s w</sub>h<sub>ose</sub> d<sub>a</sub>t<sub>a</sub> th<sub>ey pro-</sub> cess. The risks of<sub>p</sub>rivac<sub>y</sub> violations are well documented, ran<sub>g</sub>in<sub>g</sub> from reconstruction [DN03, DMT07, DY08, KRSU10], membershi<sub>p</sub>-inference [Hom+08, SOJH09, BUV14, DSSUV15, SSSS17, YGFJ18] and the unintended memorization of trainin<sub>g</sub> data [CLEKS19, Car+21, Fel20, BBFST21]. Diferential <sub>p</sub>rivac<sub>y</sub> (DP) [DMNS06] <sub>p</sub>rovides a ri<sub>g</sub>orous framework for miti<sub>g</sub>atin<sub>g</sub> these risks b<sub>y</sub> limitin<sub>g</sub> the <sub>e</sub>f<sub>ec</sub>t <sub>o</sub>f <sub>any</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l’<sub>s</sub> d<sub>a</sub>t<sub>a</sub> <sub>on</sub> th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f <sub>an</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub>’<sub>s</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>,</sub> <sub>an</sub>d h<sub>as</sub> b<sub>ecome</sub> <sub>a</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d a<sub>pp</sub>roach to <sub>p</sub>rivac<sub>y</sub> <sub>p</sub>rotection in statistics and machine learnin<sub>g</sub>.

At the same time<sub>,</sub> ensurin<sub>g</sub> <sub>p</sub>rivac<sub>y</sub> can come at a substantial cost in statistical accurac<sub>y,</sub> es<sub>p</sub>eciall<sub>y</sub> in hi<sub>g</sub>h-dimensional settin<sub>g</sub>s. This has motivated an extensive literature characterizin<sub>g</sub> the tradeofs between <sub>p</sub>rivac<sub>y,</sub> sam<sub>p</sub>le size<sub>,</sub> and accurac<sub>y,</sub> across a ran<sub>g</sub>e of hi<sub>g</sub>h-dimensional statistical and machine learnin<sub>g</sub> tasks. A central <sub>q</sub>uestion is whether <sub>p</sub>rivate al<sub>g</sub>orithms can ex<sub>p</sub>loit structure in the underl<sub>y</sub>in<sub>g</sub> distribution to reduce this cost.

Gaussian mean estimation. A natural setting in which to study these tradeofs is Gaussian mean estimation. Gi<sub>v</sub>en inde<sub>p</sub>endent sam<sub>p</sub>les from a Ga<sub>u</sub>ssian distrib<sub>u</sub>tion <sub>w</sub>ith <sub>u</sub>nkno<sub>w</sub>n mean $\mu \in \mathbb { R } ^ { d }$ and covariance matrix Σ, the <sub>g</sub>oal is to estimate <sub>�</sub> accuratel<sub>y</sub> in Euclidean distance. Mean estimation i<sub>s</sub> <sub>a</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l t<sub>as</sub>k i<sub>n</sub> it<sub>s</sub> <sub>own</sub> <sub>r</sub>i<sub>g</sub>ht <sub>an</sub>d <sub>serves</sub> <sub>as</sub> <sub>a</sub> b<sub>u</sub>ildi<sub>ng</sub> bl<sub>oc</sub>k f<sub>or</sub> <sub>more</sub> <sub>comp</sub>l<sub>ex</sub> <sub>p</sub>rocedures<sub>,</sub> includin<sub>g</sub> re<sub>g</sub>ression and o<sub>p</sub>timization. Its sim<sub>p</sub>licit<sub>y</sub> also makes it an ideal settin<sub>g</sub> in which we can isolate the inter<sub>p</sub>la<sub>y</sub> between <sub>p</sub>rivac<sub>y,</sub> dimensionalit<sub>y,</sub> and covariance structure.

Summary of our contributions. In this work, we compare the optimal error achievable for <sub>p</sub>rivate Gaussian mean estimation when the covariance is unknown to the o<sub>p</sub>timal error when the covariance is known (i.e., <sub>g</sub>iven as in<sub>p</sub>ut to the al<sub>g</sub>orithm). In the known covariance settin<sub>g</sub>, recent work [DJYZZ24] shows that the o<sub>p</sub>timal Euclidean error scales as

$$
\tilde { \Theta } \Bigg ( \sqrt { \frac { \mathrm { t r } ( \Sigma ) } { n } } + \frac { \mathrm { t r } \big ( \Sigma ^ { 1 / 2 } \big ) \sqrt { \log ( 1 / \delta ) } } { \varepsilon n } \Bigg ) ,\tag{1}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm</sub> i<sub>s</sub> th<sub>e s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>ra</sub>t<sub>e o</sub>f G<sub>auss</sub>i<sub>an mean es</sub>ti<sub>ma</sub>ti<sub>on, an</sub>d th<sub>e secon</sub>d t<sub>erm</sub> i<sub>s</sub> th<sub>e</sub> additional error due to <sub>p</sub>rivac<sub>y</sub>. Remarkabl<sub>y</sub>, the bound achieved b<sub>y</sub> [DJYZZ24] is “dimension-free”— th<sub>e error</sub> d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> th<sub>e geome</sub>t<sub>ry o</sub>f th<sub>e</sub> d<sub>a</sub>t<sub>a, an</sub>d i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f th<sub>e am</sub>bi<sub>en</sub>t di<sub>mens</sub>i<sub>on</sub> �<sub>.</sub> O<sub>ur ma</sub>i<sub>n con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons s</sub>h<sub>ow</sub> th<sub>a</sub>t thi<sub>s</sub> di<sub>mens</sub>i<sub>on-</sub>f<sub>ree</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or canno</sub>t b<sub>e recovere</sub>d i<sub>n</sub> th<sub>e un</sub>k<sub>nown</sub> covariance settin<sub>g</sub>. Our first contribution shows that for a sam<sub>p</sub>le of size <sub>�,</sub> the o<sub>p</sub>timal error in th<sub>e un</sub>k<sub>nown covar</sub>i<sub>ance se</sub>tti<sub>ng can</sub> b<sub>e</sub> l<sub>arger</sub> b<sub>y a</sub> f<sub>ac</sub>t<sub>or o</sub>f $\sqrt { d / n }$ th<sub>an</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>error</sub> i<sub>n</sub> th<sub>e</sub> known covariance settin<sub>g</sub>. Our second contribution considers the settin<sub>g</sub> when the covariance is <sub>prom</sub>i<sub>se</sub>d t<sub>o</sub> b<sub>e</sub> di<sub>agona</sub>l<sub>: we s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>even un</sub>d<sub>er</sub> thi<sub>s a</sub>dditi<sub>ona</sub>l <sub>prom</sub>i<sub>se,</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>error</sub> i<sub>n</sub> th<sub>e</sub> <sub>un</sub>k<sub>nown covar</sub>i<sub>ance se</sub>tti<sub>ng can</sub> b<sub>e</sub> l<sub>arger</sub> b<sub>y a</sub> f<sub>ac</sub>t<sub>or o</sub>f $\sqrt { d } / n ^ { 1 . 5 }$ th<sub>an</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>error</sub> i<sub>n</sub> th<sub>e</sub> k<sub>nown</sub> <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> <sub>se</sub>ttin<sub>g.</sub> T<sub>a</sub>bl<sub>e</sub> 1 <sub>su</sub>mm<sub>a</sub>riz<sub>es</sub> <sub>ou</sub>r m<sub>a</sub>in <sub>co</sub>ntrib<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub>

<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=3>Error blowup</td><td rowspan=1 colspan=1>Obstruction</td></tr><tr><td rowspan=1 colspan=1>General covariance</td><td rowspan=1 colspan=1> $\mathrm { e r r o r } _ { \mathrm { u n k n o w n } } = \tilde { \Theta } _ { \varepsilon , \delt</td><td rowspan=1 colspan=1>a } \left( 1 + \sqrt { \frac { d } { n } } \rig</td><td rowspan=1 colspan=1>ht) \cdot \mathrm { e r r o r } _ { \mathrm { k n o w n } }$ </td><td rowspan=1 colspan=1>Subspace estimation</td></tr><tr><td rowspan=1 colspan=1>Diagonal covariance</td><td rowspan=1 colspan=1> $\operatorname { e r r o r } _ { \mathrm { u n k n o w n } } = \tilde { \Theta } _ { \varepsilon , \delt</td><td rowspan=1 colspan=1>a } \left( 1 + \frac { \sqrt { d } } { n ^ { 1 . 5 } }</td><td rowspan=1 colspan=1>\right) \cdot \mathrm { e r r o r } _ { \mathrm { k n o w n } }$ </td><td rowspan=1 colspan=1>Identifying large variance coordinates</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> <sub>erro</sub> $\mathbf { \dot { \tau } } _ { \mathrm { u n k n o w n } }$ <sub>an</sub>d $\mathrm { e r r o r } _ { \mathrm { k n o w n } } ,$ <sub>respec</sub>ti<sub>ve</sub>l<sub>y,</sub> d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> E<sub>uc</sub>lid<sub>ean</sub> <sub>error</sub> i<sub>n</sub> th<sub>e</sub> <sub>un</sub>k<sub>nown</sub> <sub>an</sub>d k<sub>nown</sub> <sub>covar</sub>i<sub>ance</sub> settin<sub>g</sub>s. The known covariance bound, and the u<sub>pp</sub>er bound for unknown dia<sub>g</sub>onal covariance are due to [DJYZZ24]. The u<sub>pp</sub>er bound for unknown <sub>g</sub>eneral covariance is a folklore al<sub>g</sub>orithm based on the framework from [TCKMS22] and <sub>presen</sub>t<sub>e</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> B<sub>.</sub> Th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> $\mathrm { e r r o r } _ { \mathrm { u n k n o w n } }$ i<sub>n</sub> b<sub>o</sub>th <sub>se</sub>tti<sub>ngs are</sub> Th<sub>eorems</sub> 2<sub>.</sub>1 <sub>an</sub>d 3<sub>.</sub>2 <sub>o</sub>f thi<sub>s wor</sub>k<sub>.</sub>

O<sub>ur</sub> b<sub>oun</sub>d<sub>s</sub> hi<sub>g</sub>hli<sub>g</sub>ht t<sub>wo</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l <sub>c</sub>h<sub>a</sub>ll<sub>enges</sub> f<sub>or</sub> <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>w</sub>ith <sub>un</sub>k<sub>nown</sub> <sub>covar</sub>i<sub>ance.</sub> In the <sub>p</sub>romised dia<sub>g</sub>onal settin<sub>g,</sub> we show that ever<sub>y p</sub>rivate mean estimation al<sub>g</sub>orithm must<sub>,</sub> in <sub>some</sub> <sub>sense,</sub> “id<sub>en</sub>tif<sub>y</sub>” <sub>many</sub> <sub>o</sub>f th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>w</sub>ith th<sub>e</sub> l<sub>arges</sub>t <sub>var</sub>i<sub>ances.</sub> A<sub>n</sub>d<sub>,</sub> i<sub>n</sub> th<sub>e</sub> <sub>genera</sub>l covariance settin<sub>g,</sub> we show that ever<sub>y</sub> <sub>p</sub>rivate mean estimation al<sub>g</sub>orithm must learn the subs<sub>p</sub>ace <sub>spanne</sub>d b<sub>y</sub> th<sub>e</sub> <sub>covar</sub>i<sub>ance.</sub>

I<sub>n a</sub>dditi<sub>on</sub> t<sub>o our</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or mean es</sub>ti<sub>ma</sub>ti<sub>on</sub> i<sub>n</sub> E<sub>uc</sub>lid<sub>ean</sub> di<sub>s</sub>t<sub>ance, we s</sub>h<sub>ow</sub> th<sub>a</sub>t i<sub>n</sub> th<sub>e</sub> <sub>more c</sub>h<sub>a</sub>ll<sub>eng</sub>i<sub>ng se</sub>tti<sub>ng o</sub>f <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> i<sub>n</sub> M<sub>a</sub>h<sub>a</sub>l<sub>ano</sub>bi<sub>s error—w</sub>h<sub>ere</sub> th<sub>e error o</sub>f th<sub>e es</sub>ti<sub>ma</sub>t<sub>or</sub> i<sub>n eac</sub>h dir<sub>ec</sub>ti<sub>o</sub>n i<sub>s</sub> r<sub>equ</sub>ir<sub>e</sub>d t<sub>o</sub> b<sub>e</sub> <sub>p</sub>r<sub>opo</sub>rti<sub>o</sub>n<sub>a</sub>l t<sub>o</sub> th<sub>e</sub> <sub>va</sub>ri<sub>a</sub>n<sub>ce</sub> in th<sub>a</sub>t dir<sub>ec</sub>ti<sub>o</sub>n—<sub>p</sub>ri<sub>va</sub>t<sub>e</sub> m<sub>ea</sub>n <sub>es</sub>tim<sub>a</sub>ti<sub>o</sub>n with unknown covariance is im<sub>p</sub>ossible whenever $n \leq d - 2 $ <sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> i<sub>n s</sub>t<sub>ar</sub>k <sub>con</sub>t<sub>ras</sub>t t<sub>o</sub> th<sub>e</sub> k<sub>nown</sub> <sub>covar</sub>i<sub>ance se</sub>tti<sub>ng w</sub>h<sub>ere</sub> th<sub>e op</sub>ti<sub>ma</sub>l M<sub>a</sub>h<sub>a</sub>l<sub>ano</sub>bi<sub>s error</sub> i<sub>s</sub> fi<sub>n</sub>it<sub>e a</sub>t <sub>essen</sub>ti<sub>a</sub>ll<sub>y a</sub>ll <sub>samp</sub>l<sub>e s</sub>i<sub>zes.</sub>

## 1.1 Our results

All <sub>o</sub>f <sub>our</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> <sub>app</sub>l<sub>y</sub> t<sub>o</sub> th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> DP <sub>se</sub>tti<sub>ng</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>e</sub>l<sub>ow.</sub>

Definition 1.1 (Diferential Privacy [DMNS06]). A randomized algorithm $\mathcal { A } : X ^ { \ast }  \mathcal { W } \mathrm { i } s ( \varepsilon , \delta ) .$ diferentiall<sub>y</sub> <sub>p</sub>rivate<sub>,</sub> $\operatorname { i f } ,$ f<sub>or</sub> <sub>a</sub>ll <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> $X , X ^ { \prime } { \mathrm { , i . e . } }$ <sub>.</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> th<sub>a</sub>t dif<sub>er</sub> <sub>on</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>en</sub>t<sub>ry,</sub> <sub>an</sub>d <sub>a</sub>ll <sub>measura</sub>bl<sub>e su</sub>b<sub>se</sub>t<sub>s</sub> $W \subset \mathcal { W }$ <sub>we</sub> h<sub>ave</sub>

$$
\operatorname* { P r } [ { \mathcal { A } } ( X ) \in W ] \leq e ^ { \varepsilon } \operatorname* { P r } [ { \mathcal { A } } ( X ^ { \prime } ) \in W ] + \delta .
$$

I<sub>n or</sub>d<sub>er</sub> t<sub>o</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>e compar</sub>i<sub>son</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e error</sub> i<sub>n</sub> th<sub>e</sub> k<sub>nown an</sub>d <sub>un</sub>k<sub>nown covar</sub>i<sub>ance</sub> <sub>se</sub>tti<sub>ngs,</sub> <sub>an</sub>d t<sub>o</sub> <sub>presen</sub>t <sub>a</sub> <sub>sca</sub>l<sub>e-</sub>f<sub>ree</sub> “<sub>cos</sub>t” <sub>o</sub>f <sub>no</sub>t k<sub>now</sub>i<sub>ng</sub> th<sub>e</sub> <sub>covar</sub>i<sub>ance,</sub> <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> k<sub>nown</sub> <sub>an</sub>d <sub>u</sub>nkno<sub>w</sub>n co<sub>v</sub>ariance Ba<sub>y</sub>es risk.

Definition 1.2 (Bayes risk). Let $A _ { \varepsilon , \delta }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e se</sub>t <sub>o</sub>f $( \varepsilon , \delta ) \mathrm { - D P }$ <sub>a</sub>l<sub>gor</sub>ith<sub>ms,</sub> <sub>an</sub>d l<sub>e</sub>t $\mathcal { P }$ be a <sub>p</sub>rior di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>ove</sub>r $\pmb { \mu } \in \mathbb { R } ^ { d }$ and <sub>p</sub>ositive semi-definite $\Sigma \in \mathbb { R } ^ { d \times d }$ . The unknown-covariance Bayes risk with res<sub>p</sub>ect to the <sub>p</sub>rior $\mathcal { P }$ i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as:</sub>

$$
\Re _ { \varepsilon , \delta } ( \mathcal P , n ) = \operatorname* { i n f } \left\{ \alpha \ge 0 : \exists \mathcal { A } \in A _ { \varepsilon , \delta } \mathrm { ~ s . t . ~ } \operatorname* { P r } _ { ( \mu , \Sigma ) \sim \mathcal { P } } \left[ \left\| \mathcal { A } ( X ) - \mu \right\| _ { 2 } \le \alpha \right] \ge \frac { 3 } { 4 } \right\} .
$$

We also define the known-covariance Bayes risk (i.e., the risk when the algorithm is given the

covariance as in<sub>p</sub>ut) with res<sub>p</sub>ect to the <sub>p</sub>rior $\mathcal { P }$ <sup>b</sup><sub>y</sub>:

$$
\Re _ { \varepsilon , \delta } ^ { \mathrm { k n o w n } } ( \mathcal P , n ) = \operatorname* { i n f } \left\{ \alpha \ge 0 : \exists \mathcal { A } \in A _ { \varepsilon , \delta } \quad \mathrm { s . t . } \quad \operatorname* { P r } _ { \stackrel { ( \mu , \Sigma ) \sim \mathcal P } { X \sim N ( \mu , \Sigma ) ^ { \otimes n } } } \left[ \left\| \mathcal { A } ( X ; \Sigma ) - \mu \right\| _ { 2 } \le \alpha \right] \ge \frac 3 4 \right\} .
$$

Our results demonstrate the im<sub>p</sub>ossibilit<sub>y</sub> of <sub>p</sub>rivate mean estimation with both dimension-free error and dimension-free sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> in two settin<sub>g</sub>s. Each settin<sub>g</sub> hi<sub>g</sub>hli<sub>g</sub>hts a distinct as<sub>p</sub>ect of <sub>p</sub>rivate mean estimation that re<sub>q</sub>uires man<sub>y</sub> sam<sub>p</sub>les: In the first settin<sub>g,</sub> the variance in each basis di<sub>rec</sub>ti<sub>on</sub> i<sub>s</sub> fi<sub>xe</sub>d<sub>, an</sub>d th<sub>e</sub> difi<sub>cu</sub>lt<sub>y s</sub>t<sub>ems</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t th<sub>e</sub> b<sub>as</sub>i<sub>s</sub> di<sub>rec</sub>ti<sub>ons are c</sub>h<sub>osen un</sub>if<sub>orm</sub>l<sub>y</sub> <sub>a</sub>t <sub>ran</sub>d<sub>om, w</sub>hil<sub>e</sub> i<sub>n</sub> th<sub>e secon</sub>d <sub>se</sub>tti<sub>ng,</sub> th<sub>e</sub> b<sub>as</sub>i<sub>s</sub> di<sub>rec</sub>ti<sub>ons are</sub> fi<sub>xe</sub>d<sub>, an</sub>d th<sub>e</sub> difi<sub>cu</sub>lt<sub>y s</sub>t<sub>ems</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t th<sub>e var</sub>i<sub>ance</sub> i<sub>n eac</sub>h di<sub>rec</sub>ti<sub>on</sub> i<sub>s c</sub>h<sub>osen</sub> f<sub>rom a su</sub>it<sub>a</sub>bl<sub>y pa</sub>th<sub>o</sub>l<sub>og</sub>i<sub>ca</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub>

General covariance. In this setting, we consider mean estimation when the mean $\pmb { \mu }$ <sub>an</sub>d th<sub>e</sub> covariance � are sam<sub>p</sub>led from the <sub>p</sub>rior $\mathcal { P } _ { \mathrm { s u b } }$ d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub>

$$
( \mu , \Sigma ) \sim \mathcal { P } _ { \mathrm { s u b } } ( d , k ) \iff \left\{ \begin{array} { l l } { \mu \sim N ( 0 , d ^ { 2 } \mathbb { I } _ { d } ) } \\ { \Sigma = D D ^ { \top } \mathrm { ~ w h e r e ~ } D \sim \mathcal { H } _ { d , k } . } \end{array} \right.
$$

<sub>w</sub>h<sub>ere</sub> $\mathcal { H } _ { d , k }$ i<sub>s</sub> th<sub>e</sub> <sub>u</sub>nif<sub>o</sub>rm di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>ove</sub>r $d \times k$ <sub>ma</sub>t<sub>r</sub>i<sub>ces w</sub>ith <sub>or</sub>th<sub>onorma</sub>l <sub>co</sub>l<sub>umns.</sub> N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t th<sub>e</sub> efective dimension of ever<sub>y</sub> matrix in the su<sub>pp</sub>ort of the <sub>p</sub>rior is $k ,$ <sub>an</sub>d th<sub>a</sub>t th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> $N ( \mu , \Sigma )$ are hi<sub>g</sub>hl<sub>y</sub> structured—the data are isotro<sub>p</sub>ic in a �-dimensional subs<sub>p</sub>ace. Indeed<sub>,</sub> if � is k<sub>nown,</sub> th<sub>en</sub> th<sub>e pro</sub>bl<sub>em re</sub>d<sub>uces</sub> t<sub>o</sub> G<sub>auss</sub>i<sub>an mean es</sub>ti<sub>ma</sub>ti<sub>on w</sub>ith <sub>covar</sub>i<sub>ance</sub> $\mathbb { I } _ { k }$ <sub>.</sub> U<sub>s</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>oun</sub>d in (1) for the error in the known covariance settin<sub>g</sub> <sub>y</sub>ields

$$
\mathfrak { R } _ { \varepsilon , \delta } ^ { \mathrm { k n o w n } } ( \mathcal P _ { \mathrm { s u b } } , n ) = \tilde { O } _ { \varepsilon , \delta } \left( \sqrt { \frac { k } { n } } + \frac { k } { n } \right)\tag{2}
$$

O<sub>n</sub> th<sub>e o</sub>th<sub>er</sub> h<sub>an</sub>d<sub>, w</sub>h<sub>en</sub> � i<sub>s un</sub>k<sub>nown, a</sub> f<sub>o</sub>lkl<sub>ore a</sub>l<sub>gor</sub>ith<sub>m</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> F<sub>r</sub>i<sub>en</sub>dl<sub>y</sub>C<sub>ore</sub> f<sub>ramewor</sub>k of [TCKMS22] (<sub>p</sub>resented in Section B for com<sub>p</sub>leteness) bounds the unknown covariance risk b<sub>y</sub> $\begin{array} { r } { \tilde { O } _ { \varepsilon , \delta } \mathopen { } \mathclose \bgroup \left( \sqrt { \frac { \mathrm { t r } ( \Sigma ) } { n } } + \frac { \sqrt { d \mathrm { t r } ( \Sigma ) } } { n } \aftergroup \egroup \right) } \end{array}$ <sub>, w</sub>hi<sub>c</sub>h <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\mathfrak { R } _ { \varepsilon , \delta } ( \mathcal { P } _ { \mathrm { s u b } } , n ) = \tilde { O } _ { \varepsilon , \delta } \left( \sqrt { \frac { k } { n } } + \frac { \sqrt { d k } } { n } \right) .\tag{3}
$$

O<sub>ur</sub> <sub>ma</sub>i<sub>n</sub> <sub>resu</sub>lt <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>w</sub>h<sub>en</sub> $n \ll k$ the folklore estimator is o<sub>p</sub>timal—in <sub>p</sub>articular<sub>,</sub> a <sub>p</sub>rivate <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>canno</sub>t t<sub>a</sub>k<sub>e</sub> <sub>a</sub>d<sub>van</sub>t<sub>age</sub> <sub>o</sub>f th<sub>e</sub> l<sub>ow</sub> di<sub>mens</sub>i<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>un</sub>til it h<sub>as</sub> <sub>seen</sub> <sub>a</sub>l<sub>mos</sub>t <sub>every</sub> direction in the s<sub>p</sub>an of �.

Theorem 2.1. For all �, �, $d \in \mathbb { N }$ with $k \lesssim d$ and � log $n \lesssim k ,$ if $\varepsilon \leq 1$ and $\delta \lesssim 1 / n$ then every $( \varepsilon , \delta ) \ – D P$ algorithm M satisfies

$$
\operatorname* { P r } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { s u b } } \left[ \| \mathcal { M } ( X ) - \mu \| _ { 2 } \ge \frac { \sqrt { d k } } { n } \right] \ge \frac { 1 } { 4 } .
$$

In particular, $\begin{array} { r } { \mathfrak { R } _ { \varepsilon , \delta } ( \mathcal { P } _ { s u b } , n ) \ge \frac { \sqrt { d k } } { n } } \end{array}$

Theorem 2.1 is ti<sub>g</sub>ht in two res<sub>p</sub>ects: First, it matches the u<sub>pp</sub>er bound in (3) for the settin<sub>g</sub> <sub>w</sub>h<sub>ere</sub> <sub>�</sub> l<sub>og</sub> $n < k ,$ <sub>,</sub> <sub>an</sub>d <sub>secon</sub>d<sub>,</sub> <sub>once</sub> $n \gtrsim k + \varepsilon ^ { - 1 }$ , one can use the results of [SS21] to learn � exactl<sub>y</sub> and recover the bound in (2). Additionall<sub>y</sub>, for the settin<sub>g</sub> of � lo<sub>g</sub> $n < k .$ <sub>,</sub> com<sub>p</sub>arin<sub>g</sub> the bound in Theorem 2.1 and the bound in (2) <sub>y</sub>ields the followin<sub>g</sub> scale-free bound on the cost of unknown <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> in <sub>p</sub>ri<sub>va</sub>t<sub>e</sub> m<sub>ea</sub>n <sub>es</sub>tim<sub>a</sub>ti<sub>o</sub>n<sub>,</sub>

$$
\mathfrak { R } _ { \varepsilon , \delta } ( \mathcal { P } _ { \mathrm { s u b } } , n ) = \tilde { \Omega } _ { \varepsilon , \delta } \left( 1 + \sqrt { \frac { d } { k } } \ \right) \cdot \mathfrak { R } _ { \varepsilon , \delta } ^ { \mathrm { k n o w n } } ( \mathcal { P } _ { \mathrm { s u b } } , n ) .
$$

Settin<sub>g</sub> $k \approx n$ l<sub>og �</sub> <sub>recovers</sub> th<sub>e</sub> bl<sub>owup</sub> f<sub>or</sub> <sub>genera</sub>l <sub>covar</sub>i<sub>ance</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 1<sub>.</sub>

Diagonal covariance. While the previous setting leverages the randomness in the basis vectors, in the second settin<sub>g</sub>, we instead fix the basis vectors (i.e., we consider dia<sub>g</sub>onal covariance), and l<sub>everage</sub> th<sub>e ran</sub>d<sub>omness</sub> i<sub>n</sub> th<sub>e var</sub>i<sub>ances</sub> th<sub>emse</sub>l<sub>ves.</sub> T<sub>o</sub> d<sub>o so, we cons</sub>id<sub>er</sub> th<sub>e pr</sub>i<sub>or</sub> $\mathcal { P } _ { \mathrm { d i a g } }$ d<sub>e</sub>fi<sub>ne</sub>d <sup>b</sup><sub>y</sub>

$$
( \mu , \Sigma ) \sim \mathcal { P } _ { \mathrm { d i a g } } ( d ) \iff \left\{ \begin{array} { l l } { \mu \sim \mathcal { N } ( 0 , d ^ { 2 } \mathbb { I } _ { d } ) } \\ { \Sigma = \mathrm { D i a g } ( ( \sigma _ { i } ^ { 2 } ) _ { i \in [ d ] } ) , \mathrm { ~ w h e r e ~ } ( \sigma _ { i } ^ { 2 } ) _ { i \in [ d ] } \sim \mathrm { i n v } \chi ^ { 2 } ( 1 , 1 ) ^ { \otimes d } } \end{array} \right.
$$

where inv $\chi ^ { 2 } ( \nu , \tau ^ { 2 } )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e sca</sub>l<sub>e</sub>d<sub>-</sub>i<sub>nverse c</sub>hi<sub>-square</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on w</sub>ith <sub>�</sub> d<sub>egrees o</sub>f f<sub>ree</sub>d<sub>om</sub> and scale � (see Theorem 3.1). One can show that under the <sub>p</sub>rior $\mathcal { P } _ { \mathrm { d i a g } }$ th<sub>e covar</sub>i<sub>ance</sub> � <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\operatorname { t r } ( \Sigma ) \ \approx \ d ^ { 2 }$ <sub>an</sub>d $\mathrm { t r } ( \Sigma ^ { 1 / 2 } )$ ≈ � lo<sub>g</sub> � with hi<sub>g</sub>h constant <sub>p</sub>robabilit<sub>y</sub> (this can be formalized usin<sub>g</sub> standard bounds for sums of heav<sub>y</sub> tailed random variables, see Theorem A.3).

For the unknown dia<sub>g</sub>onal covariance settin<sub>g</sub>, an al<sub>g</sub>orithm from [DJYZZ24] <sub>g</sub>ives the <sub>g</sub>eneral b<sub>oun</sub>d $\begin{array} { r } { \tilde { O } _ { \varepsilon , \delta } \mathopen { } \mathclose \bgroup \left( \sqrt { \frac { \mathrm { t r } ( \Sigma ) } { n } } + \mathopen { } \mathclose \bgroup \left( 1 + \frac { \sqrt { d } } { n } \aftergroup \egroup \right) \frac { \mathrm { t r } ( \Sigma ^ { 1 / 2 } ) } { n } \aftergroup \egroup \right) } \end{array}$ , which combined with the bound (1) for <sub>p</sub>rivate estimation <sub>w</sub>ith k<sub>nown</sub> <sub>covar</sub>i<sub>ance,</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e</sub> k<sub>nown</sub> <sub>an</sub>d <sub>un</sub>k<sub>nown</sub> <sub>covar</sub>i<sub>ance</sub> <sub>r</sub>i<sub>s</sub>k b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> $\mathcal { P } _ { \mathrm { d i a g } } { \mathrm { : } }$

$$
\Re _ { \varepsilon , \delta } ^ { \mathrm { k n o w n } } ( \mathcal P _ { \mathrm { d i a g } } , n ) = \tilde { O } _ { \varepsilon , \delta } \bigg ( \frac { d } { \sqrt { n } } \bigg ) \qquad \mathrm { a n d } \qquad \Re _ { \varepsilon , \delta } \big ( \mathcal P _ { \mathrm { d i a g } } , n \big ) = \tilde { O } _ { \varepsilon , \delta } \bigg ( \frac { d } { \sqrt { n } } + \frac { d ^ { 3 / 2 } } { n ^ { 2 } } \bigg ) .\tag{4}
$$

O<sub>ur</sub> <sub>ma</sub>i<sub>n</sub> <sub>resu</sub>lt <sub>s</sub>h<sub>ows</sub> <sub>a</sub> ti<sub>g</sub>ht l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> <sub>un</sub>k<sub>nown-covar</sub>i<sub>ance</sub> <sub>r</sub>i<sub>s</sub>k f<sub>or</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> $\mathcal { P } _ { \mathrm { d i a g } } .$ Theorem 3.2. For all $n , d \in \mathbb { N }$ with � log $n \lesssim \sqrt { d } , i f \varepsilon \leq 1$ and $\delta \lesssim 1 / n$ then every $( \varepsilon , \delta ) \ – D P$ algorithm M satisfies

$$
\operatorname* { P r } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { d i a g } } \left[ \| \boldsymbol { M } ( \boldsymbol { X } ) - \mu \| _ { 2 } \ge \frac { d ^ { 3 / 2 } } { n ^ { 2 } \log d } \right] \ge \frac { 1 } { 4 } .
$$

In particular, $\begin{array} { r } { \mathfrak { R } _ { \varepsilon , \delta } { \left( \mathcal { P } _ { d i a g } , n \right) } \geq \frac { d ^ { 3 / 2 } } { n ^ { 2 } } } \end{array}$

As a corollar<sub>y</sub> of Theorem 3.2 and (4), we obtain the scale-free lower bound on the cost of <sub>u</sub>nkn<sub>ow</sub>n di<sub>ago</sub>n<sub>a</sub>l <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> in <sub>p</sub>ri<sub>va</sub>t<sub>e</sub> m<sub>ea</sub>n <sub>es</sub>tim<sub>a</sub>ti<sub>o</sub>n

$$
\mathfrak { R } _ { \varepsilon , \delta } \big ( \mathcal { P } _ { \mathrm { d i a g } } , n \big ) = \tilde { \Omega } _ { \varepsilon , \delta } \left( 1 + \frac { \sqrt { d } } { n ^ { 3 / 2 } } \right) \cdot \mathfrak { R } _ { \varepsilon , \delta } ^ { \mathrm { k n o w n } } \big ( \mathcal { P } _ { \mathrm { d i a g } } , n \big ) .
$$

Whil<sub>e</sub> th<sub>e</sub> bl<sub>owup</sub> i<sub>s</sub> $\Theta ( 1 )$ once $n \ge d ^ { 1 / 3 }$ <sub>, we no</sub>t<sub>e</sub> th<sub>a</sub>t th<sub>e</sub> b<sub>oun</sub>d i<sub>n</sub> Th<sub>eorem</sub> 3<sub>.</sub>2 <sub>never</sub>th<sub>e</sub>l<sub>ess</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>e necess</sub>it<sub>y o</sub>f th<sub>e a</sub>dditi<sub>ve</sub> $d ^ { 3 / 2 } / n ^ { 2 }$ t<sub>erm</sub> f<sub>or</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>reg</sub>i<sub>me</sub> <sub>w</sub>h<sub>ere</sub> <sub>�</sub> l<sub>og</sub> $n \lesssim \sqrt { d }$

T<sub>a</sub>bl<sub>e</sub> 2 <sub>summar</sub>i<sub>zes our</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or es</sub>ti<sub>ma</sub>ti<sub>on w</sub>ith E<sub>uc</sub>lid<sub>ean error an</sub>d i<sub>nc</sub>l<sub>u</sub>d<sub>es</sub> th<sub>e</sub> matchin<sub>g</sub> u<sub>pp</sub>er bounds from <sub>p</sub>rior work.
<table><tr><td>Covariance family</td><td colspan="2">Upper bound (prior work)</td><td colspan="2">Lower bound (this work)</td></tr><tr><td>Random rank-k orthogonal projector</td><td> $\tilde { O } _ { \varepsilon , \delta } \left( \sqrt { \frac { k } { n } } + \frac { \sqrt { d k } } { n } \right)$ </td><td>[TCKMS22]</td><td> $\Omega \left( { \frac { \sqrt { d k } } { n } } \right)$ </td><td>for n log r  $\iota \lesssim k$  (Theorem 2.1)</td></tr><tr><td>Inverse chi-squared diagonal prior</td><td></td><td> $\tilde { O } _ { \varepsilon , \delta } \left( \frac { d } { \sqrt { n } } + \frac { d ^ { 3 / 2 } } { n ^ { 2 } } \right) [ \mathrm { D J Y Z Z 2 4 } ]$ </td><td></td><td> $\Omega \left( { \frac { d ^ { 3 / 2 } } { n ^ { 2 } } } \right)$  for n log n  $\iota \lesssim \sqrt { d }$  (Theorem 3.2)</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> E<sub>uc</sub>lid<sub>ean upper an</sub>d l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> th<sub>e</sub> t<sub>wo</sub> h<sub>ar</sub>d <sub>covar</sub>i<sub>ance pr</sub>i<sub>ors.</sub> W<sub>e assume</sub> $\varepsilon \leq 1$ <sub>an</sub>d $\delta \lesssim 1 / n .$ <sub>.</sub> All b<sub>oun</sub>d<sub>s</sub> hold with constant <sub>p</sub>robabilit<sub>y</sub> for the corres<sub>p</sub>ondin<sub>g</sub> covariance <sub>p</sub>riors.

Estimation in Mahalanobis norm. In addition to studying private mean estimation with error in $\ell _ { 2 }$ norm<sub>,</sub> we show stron<sub>g</sub> im<sub>p</sub>ossibilit<sub>y</sub> results for <sub>p</sub>rivate mean estimation with error measured in Mahalanobis norm usin<sub>g</sub> $n \ll$ � sam<sub>p</sub>les. For a <sub>p</sub>ositive semi-definite matrix Σ and a vector <sub>�</sub>, <sub>we</sub> l<sub>e</sub>t $d _ { \Sigma } ( y )$ denote the Mahalanobis norm of<sub>�</sub> (defined in Section 4). Our main result sa<sub>y</sub>s that for $n \leq d - 2 ,$ <sup>ever</sup>y $( \varepsilon , \delta ) \mathrm { - D P }$ <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>ac</sub>hi<sub>eves</sub> fi<sub>n</sub>it<sub>e</sub> M<sub>a</sub>h<sub>a</sub>l<sub>ano</sub>bi<sub>s</sub> <sub>error</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t <sub>mos</sub>t $n \delta .$ Theorem 4.1. For all $d \in \mathbb { N }$ there exists a prior $\mathcal { P } _ { r e l }$ over instances $( \mu , \Sigma )$ such that $i f n \leq d - 2 ,$ , then for all $\varepsilon , \delta > 0$ every $( \varepsilon , \delta ) \ – D P$ algorithm M satisfies

$$
\operatorname* { P r } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { r e l } } \left[ d _ { \Sigma } ( M ( X ) - \mu ) < \infty \right] \leq n \delta .
$$

F<sub>o</sub>r $\textstyle { \delta \ < \ { \frac { 1 } { n } } . }$ <sub>,</sub> Theorem 4.1 sho<sub>w</sub>s that <sub>p</sub>ri<sub>v</sub>ate mean estimation <sub>w</sub>ith finite Mahalanobis error is <sub>essen</sub>ti<sub>a</sub>ll<sub>y</sub> i<sub>mposs</sub>ibl<sub>e w</sub>ith f<sub>ewer</sub> th<sub>an</sub> � <sub>samp</sub>l<sub>es.</sub> Thi<sub>s</sub> i<sub>s</sub> i<sub>n s</sub>t<sub>ar</sub>k <sub>con</sub>t<sub>ras</sub>t t<sub>o</sub> th<sub>e</sub> k<sub>nown covar</sub>i<sub>ance</sub> <sub>se</sub>tti<sub>ng w</sub>h<sub>ere</sub> fi<sub>n</sub>it<sub>e</sub> M<sub>a</sub>h<sub>a</sub>l<sub>ano</sub>bi<sub>s error can</sub> b<sub>e ac</sub>hi<sub>eve</sub>d f<sub>or essen</sub>ti<sub>a</sub>ll<sub>y any �.</sub> F<sub>or</sub> $\delta \ge 1 / n :$ <sub>a</sub> “<sub>name-</sub> <sub>an</sub>d<sub>-s</sub>h<sub>ame</sub>” <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> th<sub>a</sub>t <sub>pu</sub>bli<sub>s</sub>h<sub>es</sub> $\lfloor \delta n \rfloor$ <sub>recor</sub>d<sub>s se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>ran</sub>d<sub>om</sub>l<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> d<sub>a</sub>t<sub>ase</sub>t <sub>ac</sub>hi<sub>eves</sub> fi<sub>n</sub>it<sub>e</sub> Mahalanobis error<sub>,</sub> but fails most intuitive conce<sub>p</sub>ts of <sub>p</sub>rivac<sub>y</sub>.

## 1.2 Our techniques

In contrast to existin<sub>g</sub> lower bounds for <sub>p</sub>rivate Gaussian mean estimation<sub>,</sub> which <sub>p</sub>roceed b<sub>y</sub> sam<sub>p</sub>lin<sub>g</sub> $\pmb { \mu }$ f<sub>rom</sub> <sub>a</sub> <sub>su</sub>it<sub>a</sub>bl<sub>e</sub> <sub>pr</sub>i<sub>or</sub> <sub>an</sub>d <sub>samp</sub>li<sub>ng</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> f<sub>rom</sub> $\cal N ( \mu , \mathbb { I } _ { d } )$ <sub>,</sub> t<sub>o</sub> <sub>prove</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> i<sub>n</sub> <sub>our</sub> <sub>un</sub>k<sub>nown</sub> covariance settin<sub>g</sub> we must desi<sub>g</sub>n a <sub>p</sub>rior over both $\pmb { \mu }$ and �. Intuitivel<sub>y,</sub> in order to show a lower b<sub>oun</sub>d th<sub>a</sub>t <sub>excee</sub>d<sub>s</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d i<sub>n</sub> th<sub>e</sub> k<sub>nown covar</sub>i<sub>ance se</sub>tti<sub>ng,</sub> it <sub>su</sub>fi<sub>ces</sub> t<sub>o s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>an</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e o</sub>f $\pmb { \mu }$ in th<sub>e u</sub>nkn<sub>ow</sub>n <sub>cova</sub>ri<sub>a</sub>n<sub>ce se</sub>ttin<sub>g a</sub>l<sub>so co</sub>nt<sub>a</sub>in<sub>s so</sub>m<sub>e</sub> n<sub>o</sub>n-tri<sub>v</sub>i<sub>a</sub>l inf<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>b<sub>ou</sub>t �<sub>,</sub> and that learnin<sub>g</sub> this information <sub>p</sub>rivatel<sub>y</sub> re<sub>q</sub>uires man<sub>y</sub> sam<sub>p</sub>les. Indeed<sub>,</sub> a ke<sub>y</sub> ste<sub>p</sub> in our <sub>p</sub>roofs i<sub>s a re</sub>d<sub>uc</sub>ti<sub>on</sub> f<sub>rom a</sub> t<sub>as</sub>k $T ( \Sigma )$ th<sub>a</sub>t <sub>on</sub>l<sub>y</sub> i<sub>nvo</sub>l<sub>ves</sub> �<sub>—</sub>th<sub>a</sub>t i<sub>s, we</sub> l<sub>everage</sub> th<sub>e ran</sub>d<sub>omness</sub> i<sub>n</sub> $\pmb { \mu }$ to <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>a mec</sub>h<sub>an</sub>i<sub>sm w</sub>hi<sub>c</sub>h <sub>accura</sub>t<sub>e</sub>l<sub>y es</sub>ti<sub>ma</sub>t<sub>es</sub> $\pmb { \mu }$ <sub>can</sub> b<sub>e use</sub>d t<sub>o cons</sub>t<sub>ruc</sub>t <sub>an a</sub>l<sub>gor</sub>ith<sub>m w</sub>hi<sub>c</sub>h solves the task � (�). We then <sub>p</sub>rove a lower bound for <sub>p</sub>rivatel<sub>y</sub> solvin<sub>g</sub> the task � (�), which b<sub>y</sub> the reduction<sub>,</sub> im<sub>p</sub>lies a lower bound for <sub>p</sub>rivatel<sub>y</sub> estimatin<sub>g</sub> $\mu .$

General covariance. To prove our lower bound for general covariance, we define the prior over � <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub> <sub>samp</sub>l<sub>e</sub> <sub>a</sub> <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> <sub>ran</sub>d<sub>om</sub> <sub>or</sub>th<sub>onorma</sub>l $d \times k$ matrix D and let $\Sigma = D D ^ { \top } .$ <sub>—</sub>th<sub>a</sub>t i<sub>s,</sub> � is a projection matrix onto a uni<sup>f</sup>orm<sup>l</sup>y ran<sup>d</sup>om <sup>�</sup>-<sup>d</sup>imensiona<sup>l</sup> su<sup>b</sup>space. T<sup>h</sup>e <sup>k</sup>ey i<sup>d</sup>ea use<sup>d</sup> to <sub>prove</sub> th<sub>e</sub> b<sub>oun</sub>d i<sub>s</sub> th<sub>a</sub>t f<sub>or</sub> $n \leq k / 2 .$ <sub>,</sub> <sub>con</sub>diti<sub>one</sub>d <sub>on</sub> $\pmb { \mu }$ <sub>an</sub>d th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> $X = x _ { 1 } , . . . , x _ { n } \sim N ( \pmb { \mu } , \Sigma )$ the remainin<sub>g</sub> $k - n$ unseen directions in S<sub>p</sub>an(�) are uniforml<sub>y</sub> random in the remainin<sub>g</sub> $d - n$ dimensional s<sub>p</sub>ace. Intuitivel<sub>y,</sub> a <sub>p</sub>rivate mechanism must add enou<sub>g</sub>h noise to hide the contribution of a sin<sub>g</sub>le sam<sub>p</sub>le<sub>,</sub> and thus it must add noise in ever<sub>y</sub> direction in $\operatorname { S p a n } ( \Sigma )$ <sub>;</sub> h<sub>owever, s</sub>i<sub>nce con</sub>diti<sub>one</sub>d on the sam<sub>p</sub>le, the remainin<sub>g</sub> directions in S<sub>p</sub>an(�) are uniforml<sub>y</sub> random in the ambient s<sub>p</sub>ace, a <sub>p</sub>rivate al<sub>g</sub>orithm must add noise in ever<sub>y</sub> direction<sub>,</sub> which results in dimension-de<sub>p</sub>endent error.

We formalize this intuitive ar<sub>g</sub>ument b<sub>y</sub> showin<sub>g</sub> that ever<sub>y p</sub>rivate and accurate mean estimation <sub>a</sub>l<sub>gor</sub>ith<sub>m can</sub> b<sub>e use</sub>d t<sub>o pr</sub>i<sub>va</sub>t<sub>e</sub>l<sub>y genera</sub>t<sub>e a samp</sub>l<sub>e</sub> th<sub>a</sub>t <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y</sub> “<sub>corre</sub>l<sub>a</sub>t<sub>es</sub>” <sub>w</sub>ith th<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> � <sub>an</sub>d h<sub>as sma</sub>ll $\ell _ { 2 }$ <sub>norm.</sub> W<sub>e ca</sub>ll thi<sub>s samp</sub>li<sub>ng pro</sub>bl<sub>em</sub> $( \lambda , \alpha )$ <sub>-samp</sub>li<sub>ng, w</sub>h<sub>ere</sub> � <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> the “correlation” between y and �, namel<sub>y</sub> $\mathbb { E } [ \pmb { y } ^ { \top } \Sigma \pmb { y } ] \ge \lambda ^ { 2 }$ <sub>,</sub> <sub>an</sub>d <sub>�</sub> <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> th<sub>e</sub> <sub>norm,</sub> th<sub>a</sub>t i<sub>s</sub> $\| \boldsymbol { y } \| _ { 2 } \leq \alpha$ Essentiall<sub>y,</sub> our reduction shows that ever<sub>y p</sub>rivate mean estimation al<sub>g</sub>orithm with error <sub>�</sub> and sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> <sub>�</sub> can be used to construct a <sub>p</sub>rivate $( k / n , \alpha )$ -sam<sub>p</sub>ler with sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> <sub>�</sub>. To com<sub>p</sub>lete the <sub>p</sub>roof<sub>,</sub> <sub>w</sub>e sho<sub>w</sub> that e<sub>v</sub>er<sub>y</sub> <sub>p</sub>ri<sub>v</sub>ate $( \lambda , \alpha )$ -sam<sub>p</sub>ler with sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> $n \leq k / 2$ re<sub>q</sub>u<sup>i</sup>res $\alpha \geq \lambda \sqrt { d / k }$ . Plu<sub>gg</sub>in<sub>g</sub> in the <sub>p</sub>arameters from the reduction <sub>y</sub>ields that ever<sub>y</sub> <sub>p</sub>rivate mean estimation al<sub>g</sub>orithm with sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> $n \leq k / 2$ h<sub>as</sub> <sub>error</sub>

$$
\alpha \geq { \frac { \sqrt { d k } } { n } } .
$$

Sampler lower bound. To prove a lower bound for private sampling, we design a membership i<sub>n</sub>f<sub>erence</sub> <sub>a</sub>tt<sub>ac</sub>k <sub>w</sub>hi<sub>c</sub>h <sub>uses</sub> <sub>a</sub> t<sub>es</sub>t th<sub>a</sub>t di<sub>rec</sub>tl<sub>y</sub> <sub>measures</sub> th<sub>e</sub> <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>po</sub>i<sub>n</sub>t <sub>an</sub>d th<sub>e samp</sub>l<sub>e ou</sub>t<sub>pu</sub>t b<sub>y</sub> th<sub>e mec</sub>h<sub>an</sub>i<sub>sm—</sub>th<sub>a</sub>t i<sub>s,</sub> f<sub>or</sub> t<sub>es</sub>t <sub>po</sub>i<sub>n</sub>t $z \in \mathbb { R } ^ { d }$ , sam<sub>p</sub><sup>l</sup>e $\boldsymbol { y } \in \mathbb { R } ^ { d }$ <sub>, an</sub>d <sub>covar</sub>i<sub>ance</sub> $\Sigma ,$ <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> t<sub>es</sub>t

$$
\mathcal { T } _ { y , \Sigma } ( z ) = \frac { \langle z , y \rangle ^ { 2 } } { y ^ { \top } \Sigma y } .
$$

I<sub>n</sub>t<sub>u</sub>iti<sub>ve</sub>l<sub>y,</sub> if <sub>a samp</sub>l<sub>er</sub> h<sub>as seen</sub> l<sub>ess</sub> th<sub>an</sub> $k / 2$ <sub>samp</sub>l<sub>es,</sub> th<sub>en</sub> i<sub>n or</sub>d<sub>er</sub> t<sub>o ma</sub>k<sub>e</sub> $\pmb { y } ^ { \top } \pmb { \Sigma } \pmb { y }$ lar<sub>g</sub>e<sub>,</sub> it must either sam<sub>p</sub>le from a distribution with large variance in all directions, or it must make y large in the directions it has seen, which are hi<sub>g</sub>hl<sub>y</sub> de<sub>p</sub>endent on the sam<sub>p</sub>le X. Since the out<sub>p</sub>ut of a sam<sub>p</sub>ler must satisf<sub>y</sub> $\| \boldsymbol { y } \| _ { 2 } \leq \alpha ,$ the former o<sub>p</sub>tion is im<sub>p</sub>ossible, and thus, we ex<sub>p</sub>ect the out<sub>p</sub>ut y to be highl<sub>y</sub> d<sub>epen</sub>d<sub>en</sub>t <sub>on</sub> th<sub>e samp</sub>l<sub>e</sub> $\boldsymbol { X }$ <sub>.</sub> S<sub>omew</sub>h<sub>a</sub>t <sub>more</sub> f<sub>orma</sub>ll<sub>y, suppose</sub> th<sub>a</sub>t <sub>we samp</sub>l<sub>e</sub> � <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub> fi<sub>rs</sub>t <sub>samp</sub>l<sub>e</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n }$ <sub>an</sub>d l<sub>e</sub>t $P _ { X }$ denote the <sub>p</sub>rojector on S<sub>p</sub>an(X). Second, since $P _ { X }$ d<sub>e</sub>fi<sub>nes</sub> an <sub>�</sub> dimensional subs<sub>p</sub>ace<sub>,</sub> we can sam<sub>p</sub>le � b<sub>y</sub> sam<sub>p</sub>lin<sub>g</sub> the remainin<sub>g</sub> $k - n$ directions uniforml<sub>y</sub> at random from its orthogonal com<sub>p</sub>lement. We can then decom<sub>p</sub>ose the correlation between y and � into the correlation between y and $P _ { X }$ , and the correlation between y and a uniforml<sub>y</sub> random $k - n$ <sup>d</sup>imensiona<sup>l</sup> projection matrix:

$$
\begin{array} { r } { \pmb { y } ^ { \top } \pmb { \Sigma } \pmb { y } = \pmb { y } ^ { \top } P _ { \pmb { X } } \pmb { y } + \pmb { y } ^ { \top } ( \pmb { \Sigma } - P _ { \pmb { X } } ) \pmb { y } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $\left( \Sigma - P _ { \mathbf { { X } } } \right)$ is a uni<sup>f</sup>orm<sup>l</sup>y ran<sup>d</sup>om projection <sup>f</sup>rom a $d - n$ dimensional s<sub>p</sub>ace to a �−� dimensional s<sub>p</sub>ace. Thus using the norm bound on y, we have that $\pmb { y } ^ { \top } ( \pmb { \Sigma } - \pmb { P } _ { \pmb { X } } ) \pmb { y }$ is t<sub>yp</sub>icall<sub>y</sub> at most $\alpha ^ { 2 } k / d .$ Si<sub>nce</sub> th<sub>e</sub> l<sub>e</sub>ft h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> $( \pmb { y } ^ { \top } \pmb { \Sigma } \pmb { y } )$ i<sub>s</sub> in <sub>e</sub>x<sub>pec</sub>t<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>t l<sub>eas</sub>t $\lambda ^ { 2 }$ <sub>, we see</sub> th<sub>a</sub>t $y ^ { \top } P _ { X } y \ge \lambda ^ { 2 } / 2$ <sub>w</sub>h<sub>enever</sub> $\alpha \leq \lambda \sqrt { d / 2 k }$ <sub>.</sub> Th<sub>us,</sub> <sub>a</sub>n<sub>y</sub> $( \lambda , \alpha )$ <sub>-samp</sub>l<sub>er</sub> <sub>w</sub>ith $\alpha \leq \lambda \sqrt { d / 2 k }$ must exhibit si<sub>g</sub>nificant correlation with the data X. Levera<sub>g</sub>in<sub>g</sub> this correlation allows us to show that the attack $\mathcal { T }$ can distin<sub>g</sub>uish a member in X from a fresh sam<sub>p</sub>le from the <sub>p</sub>o<sub>p</sub>ulation, and thus, such a $( \lambda , \alpha )$ -sam<sub>p</sub>ler cannot be <sub>p</sub>rivate.

Diagonal covariance. A first step in proving the lower bound in the diagonal covariance setting is constructin<sub>g</sub> a <sub>p</sub>rior over � (or famil<sub>y</sub> of covariance matrices) for which we ex<sub>p</sub>ect <sub>p</sub>rivate mean <sub>es</sub>ti<sub>ma</sub>ti<sub>on w</sub>ith <sub>un</sub>k<sub>nown covar</sub>i<sub>ance</sub> t<sub>o</sub> b<sub>e muc</sub>h h<sub>ar</sub>d<sub>er</sub> th<sub>an pr</sub>i<sub>va</sub>t<sub>e es</sub>ti<sub>ma</sub>ti<sub>on w</sub>ith k<sub>nown co-</sub> variance. Our startin<sub>g p</sub>oint is the al<sub>g</sub>orithm of [DJYZZ24] which <sub>p</sub>roceeds as follows: First learn th<sub>e coor</sub>di<sub>na</sub>t<sub>es w</sub>ith th<sub>e</sub> $k \approx n ^ { 2 }$ lar<sub>g</sub>est variances<sub>,</sub> and estimate the mean in those coordinates usin<sub>g</sub> th<sub>e</sub> k<sub>nown</sub> <sub>covar</sub>i<sub>ance</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m;</sub> <sub>an</sub>d <sub>secon</sub>d<sub>,</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> th<sub>e</sub> <sub>mean</sub> i<sub>n</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> $d - k$ <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> b<sub>y a</sub>ddi<sub>ng sp</sub>h<sub>er</sub>i<sub>ca</sub>l <sub>no</sub>i<sub>se propor</sub>ti<sub>ona</sub>l t<sub>o</sub> th<sub>e</sub> t<sub>race.</sub> At <sub>a</sub> hi<sub>g</sub>h l<sub>eve</sub>l<sub>, one can s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>error</sub> <sub>o</sub>f thi<sub>s</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>compare</sub>d t<sub>o</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l k<sub>nown</sub> <sub>covar</sub>i<sub>ance</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> i<sub>s</sub> l<sub>arges</sub>t <sub>w</sub>h<sub>en</sub> th<sub>e</sub> dia<sub>g</sub>onal entries $\sigma _ { i } ^ { 2 }$ <sup>d</sup>eca<sub>y</sub> <sub>p</sub>ro<sub>p</sub>ort<sup>i</sup>ona<sup>l</sup> to $i ^ { - 2 }$ <sub>.</sub> Thi<sub>s</sub> l<sub>ea</sub>d<sub>s</sub> <sub>us</sub> t<sub>o</sub> <sub>samp</sub>l<sub>e</sub> <sub>eac</sub>h $\pmb { \sigma } _ { i } ^ { 2 }$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> f<sub>rom</sub> the scaled-inverse-chi-squared distribution inv $\chi ^ { 2 } ( 1 , 1 )$ <sub>.</sub> Thi<sub>s</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> h<sub>as a</sub> t<sub>a</sub>il <sub>o</sub>f <sub>or</sub>d<sub>er</sub> $t ^ { - 1 / 2 }$ , so it<sub>s</sub> �<sub>-</sub>th l<sub>arges</sub>t d<sub>raw</sub> i<sub>s</sub> t<sub>yp</sub>i<sub>ca</sub>ll<sub>y on</sub> th<sub>e sca</sub>l<sub>e</sub> $d ^ { 2 } i ^ { - 2 }$

N<sub>ex</sub>t<sub>,</sub> t<sub>o prove our</sub> l<sub>ower</sub> b<sub>oun</sub>d f<sub>or</sub> di<sub>agona</sub>l <sub>covar</sub>i<sub>ance, we s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>every pr</sub>i<sub>va</sub>t<sub>e an</sub>d <sub>accura</sub>t<sub>e</sub> <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>can</sub> b<sub>e</sub> <sub>use</sub>d t<sub>o</sub> <sub>pr</sub>i<sub>va</sub>t<sub>e</sub>l<sub>y</sub> <sub>se</sub>l<sub>ec</sub>t th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>w</sub>ith th<sub>e</sub> � l<sub>arges</sub>t <sub>var</sub>i<sub>ances—</sub> that is, we reduce from a “to<sub>p</sub>-�” variance selection <sub>p</sub>roblem that we call (�, ℓ)-selection. In $( k , \ell )$ selection we are <sub>g</sub>iven sam<sub>p</sub>les $X \ = \ x _ { 1 } , . . . , x _ { n } \ \sim \ N ( 0 , \Sigma ) ^ { \otimes n }$ <sub>w</sub>h<sub>ere</sub> $\Sigma \ = \ \mathrm { D i a g } ( \pmb { \sigma } _ { 1 } ^ { 2 } , . . . , \pmb { \sigma } _ { d } ^ { 2 } ) \ \sim$ $\mathsf { i n v } \chi ^ { 2 } ( 1 , 1 ) ^ { \otimes d }$ <sub>, an</sub>d th<sub>e goa</sub>l i<sub>s</sub> t<sub>o ou</sub>t<sub>pu</sub>t <sub>a</sub> li<sub>s</sub>t <sub>o</sub>f <sub>a</sub>t <sub>mos</sub>t ℓ <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> th<sub>a</sub>t <sub>con</sub>t<sub>a</sub>i<sub>ns a</sub>t l<sub>eas</sub>t $1 / 4$ <sub>o</sub>f th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es w</sub>ith th<sub>e</sub> �<sub>-</sub>l<sub>arges</sub>t <sub>var</sub>i<sub>ances.</sub> F<sub>orma</sub>ll<sub>y, our re</sub>d<sub>uc</sub>ti<sub>on s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>every pr</sub>i<sub>va</sub>t<sub>e mean</sub> estimation al<sub>g</sub>orithm that achieves error $\alpha$ <sub>can</sub> b<sub>e use</sub>d t<sub>o pr</sub>i<sub>va</sub>t<sub>e</sub>l<sub>y se</sub>l<sub>ec</sub>t th<sub>e coor</sub>di<sub>na</sub>t<sub>es w</sub>ith th<sub>e</sub> � l<sub>arges</sub>t <sub>var</sub>i<sub>ances,</sub> <sub>w</sub>hil<sub>e</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>ng</sub> <sub>a</sub>t <sub>mos</sub>t $\begin{array} { r } { \ell \approx \frac { \alpha ^ { 2 } k n ^ { 2 } } { d ^ { 2 } } } \end{array}$ <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>overa</sub>ll<sub>.</sub> W<sub>e</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e</sub> <sub>proo</sub>f b<sub>y</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> <sub>a</sub> l<sub>ower</sub> b<sub>oun</sub>d f<sub>or</sub> <sub>pr</sub>i<sub>va</sub>t<sub>e</sub> $( k , \ell )$ selection which states that an<sub>y</sub> <sub>p</sub>rivate $( k , \ell )$ <sub>-se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m w</sub>ith $n \leq { \sqrt { k } }$ sam<sub>p</sub><sup>l</sup>es re<sub>q</sub>u<sup>i</sup>res $\ell \geq d / { \log d } .$ Settin<sub>g</sub> $k \approx n ^ { 2 }$ <sub>an</sub>d <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e se</sub>l<sub>ec</sub>ti<sub>on</sub> lower bound with the settin<sub>g</sub> of ℓ in the reduction<sub>,</sub> we obtain that ever<sub>y</sub> <sub>p</sub>rivate mean estimation <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> h<sub>as</sub> <sub>error</sub>

$$
\alpha \ge \frac { d ^ { 3 / 2 } } { n \sqrt { k } } \approx \frac { d ^ { 3 / 2 } } { n ^ { 2 } } .
$$

Selection lower bound. To show a lower bound for private selection we construct a membershi<sub>p</sub> inference attack a<sub>g</sub>ainst an<sub>y</sub> $( k , \ell )$ <sub>-se</sub>l<sub>ec</sub>ti<sub>on a</sub>l<sub>gor</sub>ith<sub>m</sub> th<sub>a</sub>t <sub>uses</sub> $n \leq { \sqrt { k } }$ <sub>samp</sub>l<sub>es an</sub>d <sub>a</sub>t th<sub>e</sub> <sub>same</sub> ti<sub>me</sub> <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>a</sub>t <sub>mos</sub>t $\ell \leq d / \log d$ <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> O<sub>ne na</sub>t<sub>ura</sub>l <sub>can</sub>did<sub>a</sub>t<sub>e</sub> f<sub>or suc</sub>h <sub>an a</sub>tt<sub>ac</sub>k<sub>,</sub> i<sub>nsp</sub>i<sub>re</sub>d b<sub>y</sub> [SU17] is the followin<sub>g</sub> test: For test <sub>p</sub>oint $z \in \mathbb { R } ^ { d }$ , covariance $\Sigma ,$ <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>ti<sub>on vec</sub>t<sub>or</sub> $s \in \{ 0 , 1 \} ^ { d }$ d<sub>e</sub>fi<sub>ne</sub>

$$
{ \mathcal { T } } _ { \mathrm { n a i v e } } ( z , s , \Sigma ) : = \sum _ { j \in [ d ] } \left( { \frac { z _ { j } ^ { 2 } } { \sigma _ { j } ^ { 2 } } } - 1 \right) s _ { j } .
$$

One mi<sub>g</sub>ht ho<sub>p</sub>e that when the test <sub>p</sub>oint ${ \mathbf { } } x _ { i }$ is in the dataset the ex<sub>p</sub>ectation $\mathbb { E } [ \mathcal { T } _ { \mathrm { n a i v e } } ( \pmb { x } _ { i } ) ]$ <sub>w</sub>ill b<sub>e</sub> large. Unfortunately, this is not the case in general: Using properties of the inv $\chi ^ { 2 }$ <sub>p</sub>r<sup>i</sup>or on $\Sigma ,$ one <sub>can s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>con</sub>diti<sub>ona</sub>l <sub>on</sub> $\boldsymbol { X }$ <sub>,</sub> th<sub>e expec</sub>t<sub>a</sub>ti<sub>on o</sub>f <sub>eac</sub>h t<sub>erm</sub> i<sub>n</sub> $\mathcal { T } _ { \mathrm { n a i v e } }$ i<sub>s</sub> $\frac { { { q } _ { j } } - 1 } { 1 + n { { q } _ { j } } }$ <sub>, w</sub>h<sub>ere</sub> $\mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta \delta } }$ d<sub>eno</sub>t<sub>es</sub> t<sup>h</sup>e empirica<sup>l</sup> variance in coor<sup>d</sup>inate �. T<sup>h</sup>us, w<sup>h</sup>en $q _ { j } \gg 1$ , t<sup>h</sup>e contri<sup>b</sup>ution o<sup>f</sup> term � is a<sup>b</sup>out $1 / n$ b<sub>u</sub>t <sub>w</sub>h<sub>en</sub> $\mathbf { \Delta } q _ { j } \in ( 0 , 1 )$ the contribution is ne<sub>g</sub>ative and is t<sub>yp</sub>icall<sub>y</sub> on the order of $\dot { \mathbf { \zeta } } - 1 / n .$ It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> t<sub>op</sub> �<sub>-</sub>t<sub>erms may con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> $k / n$ t<sub>o</sub> th<sub>e</sub> t<sub>es</sub>t<sub>, a care</sub>f<sub>u</sub>l <sub>se</sub>l<sub>ec</sub>ti<sub>on a</sub>l<sub>gor</sub>ith<sub>m may se</sub>l<sub>ec</sub>t <sub>many o</sub>f th<sub>e coor</sub>di<sub>na</sub>t<sub>es w</sub>ith th<sub>e sma</sub>ll<sub>es</sub>t <sub>var</sub>i<sub>ances, an</sub>d th<sub>ese may con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e up</sub> t<sub>o</sub> $- \ell / n$ t<sub>o</sub> th<sub>e</sub> t<sub>es</sub>t<sub>.</sub> H<sub>ence,</sub> thi<sub>s approac</sub>h <sub>can on</sub>l<sub>y</sub> h<sub>ope</sub> t<sub>o s</sub>h<sub>ow</sub> th<sub>a</sub>t ℓ <sub>mus</sub>t b<sub>e a</sub>t l<sub>eas</sub>t $k ,$ <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> t<sub>oo wea</sub>k f<sub>or</sub> our <sub>p</sub>ur<sub>p</sub>oses (we aim to show that ℓ must be at least $d / \log d$ <sub>even w</sub>h<sub>en</sub> $k \approx n ^ { 2 } )$ <sub>.</sub> Th<sub>e pro</sub>bl<sub>ema</sub>ti<sub>c</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> i<sub>n</sub> th<sub>e</sub> t<sub>es</sub>t $\mathcal { T } _ { \mathrm { n a i v e } }$ <sub>occurs</sub> b<sub>ecause</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>es w</sub>ith th<sub>e</sub> t<sub>op-</sub>� <sub>var</sub>i<sub>ances</sub> d<sub>on</sub>’t <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> <sub>enoug</sub>h t<sub>o</sub> th<sub>e sum</sub> t<sub>o overcome</sub> th<sub>e many sma</sub>ll <sub>nega</sub>ti<sub>ve con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> f<sub>rom</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>es w</sub>h<sub>ere</sub> $\sigma _ { j } ^ { 2 } \in ( 0 , 1 )$ <sub>.</sub> Th<sub>us, a na</sub>t<sub>ura</sub>l <sub>way</sub> t<sub>o c</sub>i<sub>rcumven</sub>t thi<sub>s</sub> b<sub>arr</sub>i<sub>er</sub> i<sub>s</sub> t<sub>o sca</sub>l<sub>e eac</sub>h t<sub>erm</sub> b<sub>y</sub> $\sigma _ { j }$ <sub>.</sub> B<sub>ecause</sub> th<sub>e</sub> <sub>sca</sub>li<sub>ng</sub> i<sub>s</sub> <sub>asymme</sub>t<sub>r</sub>i<sub>c,</sub> th<sub>e</sub> <sub>resu</sub>lt i<sub>s</sub> th<sub>a</sub>t th<sub>e</sub> l<sub>arge</sub> t<sub>erms</sub> b<sub>ecome</sub> <sub>muc</sub>h l<sub>arger,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> i<sub>mpac</sub>t <sub>on</sub> th<sub>e</sub> <sub>sma</sub>ll t<sub>erms</sub> i<sub>s</sub> <sub>neg</sub>li<sub>g</sub>ibl<sub>e.</sub> Thi<sub>s</sub> l<sub>ea</sub>d<sub>s</sub> <sub>us</sub> t<sub>o</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>more</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> t<sub>es</sub>t

$$
\mathcal { T } _ { s , \Sigma } ( z ) : = \sum _ { j \in [ d ] } \left( \frac { z _ { j } ^ { 2 } } { \sigma _ { j } ^ { 2 } } - 1 \right) \operatorname* { m i n } \biggl \{ \sigma _ { j } , \frac { d } { k } \biggr \} s _ { j } ,
$$

where the min is re<sub>q</sub>uired for technical reasons (the $\sigma _ { j } ^ { 2 } { } ^ { , }$ <sub>s</sub> <sub>are</sub> h<sub>eavy</sub> t<sub>a</sub>il<sub>e</sub>d<sub>,</sub> <sub>an</sub>d th<sub>us</sub> <sub>c</sub>li<sub>pp</sub>i<sub>ng</sub> h<sub>e</sub>l<sub>ps</sub> t<sub>o</sub> bound the variance). Using properties of the inv $\chi ^ { 2 } ( 1 , 1 )$ di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on, one can s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> t<sub>op-</sub>� <sub>var</sub>i<sub>ances</sub> <sub>are</sub> <sub>a</sub>t l<sub>eas</sub>t $( d / k ) ^ { 2 }$ <sub>,</sub> <sub>an</sub>d th<sub>us,</sub> th<sub>e</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> t<sub>erms</sub> i<sub>n</sub> $\mathcal { T }$ in<sub>c</sub>r<sub>eases</sub> b<sub>y</sub> <sub>a</sub> f<sub>ac</sub>t<sub>or</sub> <sub>o</sub>f $d / k$ <sub>w</sub>h<sub>en</sub> <sub>compare</sub>d t<sub>o</sub> $\mathcal { T } _ { \mathrm { n a i v e } }$ . Since the ne<sub>g</sub>ative terms are essentiall<sub>y</sub> unafected (the<sub>y</sub> h<sub>ave</sub> <sub>sma</sub>ll $\sigma _ { j } )$ , we on<sup>l</sup><sub>y</sub> re<sub>q</sub>u<sup>i</sup>re $\ell \leq d / \log c$ to <sub>g</sub>uarantee t<sup>h</sup>at $\mathbb { E } [ \mathcal { T } ( \pmb { x } _ { i } ) ]$ is lar<sub>g</sub>e.

## 1.3 Related work

Diferentially private Gaussian mean estimation. Diferentially private mean estimation has been studied extensivel<sub>y</sub>, includin<sub>g</sub> as<sub>y</sub>m<sub>p</sub>toticall<sub>y</sub> eficient estimators [Smi11] and estimators for univariate Gaussians with unknown <sub>p</sub>arameters [KV18]. A series of works established nearl<sub>y</sub> optimal guarantees for Gaussian mean estimation with known covariance in Mahalanobis error metric [BKSW19, KLSU19, AAK21, LKKO21]. Brown et al. [BGSUZ21] develo<sub>p</sub>ed o<sub>p</sub>timal Gaussian <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>ors</sub> i<sub>n</sub> M<sub>a</sub>h<sub>a</sub>l<sub>ano</sub>bi<sub>s</sub> <sub>error</sub> <sub>me</sub>t<sub>r</sub>i<sub>c</sub> <sub>w</sub>ith <sub>un</sub>k<sub>nown</sub> <sub>covar</sub>i<sub>ance,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l time al<sub>g</sub>orithms matchin<sub>g</sub> these <sub>g</sub>uarantees [BHS23, KDH23]. As defined above, this loss is afine in<sub>va</sub>ri<sub>a</sub>nt <sub>a</sub>nd im<sub>p</sub>li<sub>es</sub>

$$
\| \tilde { \mu } - \mu \| _ { 2 } \leq d _ { \Sigma } ( \tilde { \mu } - \mu ) \sqrt { \| \Sigma \| _ { \mathrm { o p } } } .
$$

Ho<sub>w</sub>e<sub>v</sub>er<sub>,</sub> it normalizes errors b<sub>y</sub> the <sub>v</sub>ariance in each direction<sub>,</sub> incl<sub>u</sub>din<sub>g</sub> directions <sub>w</sub>ith <sub>v</sub>er<sub>y</sub> <sub>sma</sub>ll <sub>var</sub>i<sub>ance,</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> d<sub>oes</sub> <sub>no</sub>t <sub>cap</sub>t<sub>ure</sub> th<sub>e</sub> f<sub>u</sub>ll b<sub>ene</sub>fit<sub>s</sub> <sub>o</sub>f <sub>an</sub>i<sub>so</sub>t<sub>ropy</sub> <sub>un</sub>d<sub>er</sub> E<sub>uc</sub>lid<sub>ean</sub> error. Moreover, the sam<sub>p</sub>le size re<sub>q</sub>uired b<sub>y</sub> the methods in [BGSUZ21] is at least $n \gtrsim d / \varepsilon .$ <sub>.</sub> O<sub>u</sub>r <sub>comp</sub>l<sub>emen</sub>t<sub>ary resu</sub>lt<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>a samp</sub>l<sub>e s</sub>i<sub>ze o</sub>f $n \gtrsim d$ is re<sub>q</sub>uired to achieve an<sub>y</sub> finite error in Mahalanobis distance<sub>,</sub> showin<sub>g</sub> that this re<sub>q</sub>uirement is almost ti<sub>g</sub>ht.

For Euclidean error, the Friendl<sub>y</sub>Core framework [TCKMS22] <sub>p</sub>rovides a filterin<sub>g</sub>-based a<sub>pp</sub>roach to <sub>p</sub>rivate a<sub>gg</sub>re<sub>g</sub>ation and underlies the <sub>g</sub>eneral-<sub>p</sub>ur<sub>p</sub>ose Gaussian mean estimator used as a benchmark in this work. Aimin<sub>g</sub> to ex<sub>p</sub>loit covariance structure, Aumüller et al. [ALNP24] develo<sub>p</sub>ed variance-aware <sub>p</sub>rivate mean estimators for a<sub>pp</sub>roximatel<sub>y</sub>-known dia<sub>g</sub>onal covariance matrices<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>ac</sub>hi<sub>eve</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>error</sub> <sub>as</sub> l<sub>ong</sub> <sub>as</sub> $n \gtrsim \sqrt { d } / \varepsilon$ . Da<sub>g</sub>an et al. [DJYZZ24] established o<sub>p</sub>timal dimension-free <sub>g</sub>uarantees u<sub>p</sub> to lo<sub>g</sub>arithmic factors when the covariance is known<sub>,</sub> and im<sub>p</sub>roved dimension de<sub>p</sub>endence when it is dia<sub>g</sub>onal and unknown. Under <sub>p</sub>ure diferential <sub>p</sub>rivac<sub>y,</sub> Chen et al. [CDMM26] also obtain covariance-de<sub>p</sub>endent Euclidean <sub>g</sub>uarantees for known-covariance Gaussian m<sub>ea</sub>n <sub>es</sub>tim<sub>a</sub>ti<sub>o</sub>n<sub>,</sub> in <sub>a</sub> r<sub>eg</sub>im<sub>e</sub> r<sub>equ</sub>irin<sub>g</sub> <sub>a</sub>t l<sub>eas</sub>t $d / \varepsilon$ sam<sub>p</sub><sup>l</sup>es.

Outside of diferential <sub>p</sub>rivac<sub>y</sub>, [MZ25] and [DL22] established o<sub>p</sub>timal, dimension-free robust estimators for the mean of anisotro<sub>p</sub>ic Gaussian distributions under <sub>g</sub>eneral norms.

Privately learning covariance structure. Privately learning the covariance or a low-dimensional <sub>su</sub>b<sub>space can ena</sub>bl<sub>e su</sub>b<sub>sequen</sub>t <sub>mean es</sub>ti<sub>ma</sub>ti<sub>on w</sub>ith f<sub>ewer samp</sub>l<sub>es or</sub> b<sub>e</sub>tt<sub>er accuracy,</sub> b<sub>u</sub>t th<sub>ese</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e</sub> t<sub>as</sub>k<sub>s can</sub> th<sub>emse</sub>l<sub>ves requ</sub>i<sub>re a</sub> l<sub>o</sub>t <sub>o</sub>f d<sub>a</sub>t<sub>a.</sub>

Lower bounds for <sub>p</sub>rivate Gaussian covariance estimation were established b<sub>y</sub> [KMS22] and stren<sub>g</sub>thened b<sub>y</sub> [Nar25, PH25]. These results show that learnin<sub>g</sub> the covariance of a Gaussian di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>w</sub>ith <sub>cons</sub>t<sub>an</sub>t <sub>spec</sub>t<sub>ra</sub>l <sub>accuracy,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>wou</sub>ld <sub>su</sub>fi<sub>ce</sub> f<sub>or</sub> <sub>su</sub>b<sub>sequen</sub>tl<sub>y</sub> <sub>us</sub>i<sub>ng</sub> <sub>mean</sub> estimation al<sub>g</sub>orithms <sub>w</sub>ith kno<sub>w</sub>n co<sub>v</sub>ariance<sub>,</sub> re<sub>qu</sub>ires $n \gtrsim d ^ { 3 / 2 }$ sam<sub>p</sub><sup>l</sup>es.

Sin<sub>g</sub>hal and Steinke [SS21] <sub>g</sub>ive dimension-inde<sub>p</sub>endent sam<sub>p</sub>le bounds for exact recover<sub>y</sub> of a subs<sub>p</sub>ace under a <sub>g</sub>eneral <sub>p</sub>osition assum<sub>p</sub>tion<sub>,</sub> as well as a<sub>pp</sub>roximate recover<sub>y</sub> under a suficientl<sub>y</sub> lar<sub>g</sub>e ei<sub>g</sub>envalue <sub>g</sub>a<sub>p</sub>. Tsfadia [Tsf24] studies subs<sub>p</sub>ace estimation in a distribution-free settin<sub>g</sub> and f<sub>ur</sub>th<sub>er</sub> <sub>quan</sub>tifi<sub>es</sub> th<sub>e</sub> <sub>ro</sub>l<sub>e</sub> <sub>o</sub>f <sub>e</sub>i<sub>genva</sub>l<sub>ue</sub> <sub>gaps.</sub>

R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d b<sub>a</sub>rri<sub>e</sub>r<sub>s</sub> <sub>a</sub>ri<sub>se</sub> in <sub>p</sub>ri<sub>va</sub>t<sub>e</sub> <sub>spa</sub>r<sub>se</sub> <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> <sub>es</sub>tim<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd PCA<sub>,</sub> <sub>w</sub>h<sub>e</sub>r<sub>e</sub> r<sub>ow</sub>-<sub>co</sub>l<sub>u</sub>mn <sub>spars</sub>it<sub>y</sub> <sub>o</sub>f th<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>a</sub>l<sub>one</sub> <sub>nee</sub>d <sub>no</sub>t <sub>e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>e</sub> th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l d<sub>epen</sub>d<sub>ence</sub> <sub>o</sub>f th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> size on the ambient dimension [KPST26]. Our lower bounds establish the cost of unknown covariance directl<sub>y</sub> for mean estimation<sub>,</sub> without re<sub>q</sub>uirin<sub>g</sub> the estimator to recover the covariance or its <sub>p</sub>rinci<sub>p</sub>al dir<sub>ec</sub>ti<sub>o</sub>n<sub>s.</sub>

Fingerprinting lower bounds. Our lower bounds build on the intuitive connection between accurate estimation and membershi<sub>p</sub> inference. Bun, Ullman, and Vadhan [BUV14] introduced fin-<sub>g</sub>er<sub>p</sub>rintin<sub>g</sub> codes as a tool for information-theoretic lower bounds in diferential <sub>p</sub>rivac<sub>y,</sub> and Dwork et al. [DSSUV15] develo<sub>p</sub>ed robust tracin<sub>g</sub> attacks for binar<sub>y</sub> and Gaussian data. These techni<sub>q</sub>ues were further develo<sub>p</sub>ed for Gaussian estimation b<sub>y</sub> Kamath et al. [KLSU19], for ex<sub>p</sub>onential families b<sub>y</sub> Kamath, Mouzakis, and Sin<sub>g</sub>hal [KMS22], and for <sub>g</sub>eneral <sub>p</sub>arametric models throu<sub>g</sub>h score attacks b<sub>y</sub> Cai, Wan<sub>g</sub>, and Zhan<sub>g</sub> [CWZ23]. Peter, Tsfadia, and Ullman [PTU24] develo<sub>p</sub> <sub>p</sub>addin<sub>g</sub>- and-<sub>p</sub>ermutin<sub>g</sub> fin<sub>g</sub>er<sub>p</sub>rintin<sub>g</sub> codes that var<sub>y</sub> smoothl<sub>y</sub> with the re<sub>q</sub>uired accurac<sub>y</sub>.

Our tracin<sub>g</sub> ar<sub>g</sub>uments ex<sub>p</sub>loit <sub>p</sub>riors over the covariance whose conditional distributions <sub>g</sub>iven the sam<sub>p</sub>le can be anal<sub>y</sub>zed ex<sub>p</sub>licitl<sub>y</sub>. The additional challen<sub>g</sub>e in our settin<sub>g</sub> is to ex<sub>p</sub>loit uncertaint<sub>y</sub> a<sup>b</sup>out t<sup>h</sup>e covariance: we <sup>d</sup>esign joint priors over t<sup>h</sup>e mean an<sup>d</sup> covariance, re<sup>d</sup>uce mean estimation t<sub>o</sub> <sub>covar</sub>i<sub>ance-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>samp</sub>li<sub>ng</sub> <sub>or</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> t<sub>as</sub>k<sub>s,</sub> <sub>an</sub>d <sub>cons</sub>t<sub>ruc</sub>t t<sub>rac</sub>i<sub>ng</sub> <sub>a</sub>tt<sub>ac</sub>k<sub>s</sub> f<sub>or</sub> th<sub>ese</sub> t<sub>as</sub>k<sub>s</sub> i<sub>ns</sub>t<sub>ea</sub>d<sub>.</sub>

Private selection. Our diagonal-covariance lower bound is particularly related to the top-� selection <sub>p</sub>roblem of Steinke and Ullman [SU17], who extended fin<sub>g</sub>er<sub>p</sub>rintin<sub>g</sub> ar<sub>g</sub>uments to selectin<sub>g</sub> coordinates with lar<sub>g</sub>e means. Related <sub>q</sub>uestions arise in <sub>p</sub>rivate s<sub>p</sub>arse mean estimation [CWZ19]. Our auxiliar<sub>y</sub> task, called (�, ℓ)-selection, instead asks for a list of len<sub>g</sub>th ℓ containin<sub>g</sub> a constant f<sub>rac</sub>ti<sub>on o</sub>f th<sub>e coor</sub>di<sub>na</sub>t<sub>es w</sub>ith th<sub>e</sub> l<sub>arges</sub>t <sub>var</sub>i<sub>ances, an</sub>d <sub>a</sub>ll<sub>ows</sub> th<sub>e</sub> li<sub>s</sub>t t<sub>o con</sub>t<sub>a</sub>i<sub>n a</sub> l<sub>o</sub>t <sub>more</sub> th<sub>an</sub> � <sub>coo</sub>rdin<sub>a</sub>t<sub>es.</sub> A dir<sub>ec</sub>t <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> m<sub>ea</sub>n-<sub>se</sub>l<sub>ec</sub>ti<sub>o</sub>n tr<sub>ac</sub>in<sub>g a</sub>r<sub>gu</sub>m<sub>e</sub>nt i<sub>s</sub> n<sub>o</sub>t <sub>su</sub>fi<sub>c</sub>i<sub>e</sub>nt f<sub>o</sub>r <sub>ou</sub>r <sub>purposes: con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> f<sub>rom se</sub>l<sub>ec</sub>t<sub>e</sub>d l<sub>ow-var</sub>i<sub>ance coor</sub>di<sub>na</sub>t<sub>es can cance</sub>l th<sub>ose</sub> f<sub>rom</sub> hi<sub>g</sub>h<sub>-var</sub>i<sub>ance</sub> coordinates. We address this issue usin<sub>g</sub> a wei<sub>g</sub>hted<sub>,</sub> truncated test<sub>,</sub> ada<sub>p</sub>ted to our <sub>p</sub>rior on the <sub>va</sub>ri<sub>a</sub>n<sub>ces.</sub>

## 2 Lower bound for mean estimation with general covariance

Thi<sub>s</sub> <sub>sec</sub>ti<sub>on</sub> <sub>prov</sub>id<sub>es</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>o</sub>f th<sub>e</sub> B<sub>ayes</sub> <sub>r</sub>i<sub>s</sub>k f<sub>or</sub> <sub>ran</sub>d<sub>om</sub> <sub>non-</sub>di<sub>agona</sub>l <sub>covar</sub>i<sub>ance</sub> <sub>ma</sub>t<sub>r</sub>i<sub>ces,</sub> uni<sup>f</sup>orm<sup>l</sup>y <sup>d</sup>istri<sup>b</sup>ute<sup>d</sup> among t<sup>h</sup>e ort<sup>h</sup>ogona<sup>l</sup> projectors on <sup>�</sup>-<sup>d</sup>imensiona<sup>l</sup> <sup>l</sup>inear su<sup>b</sup>spaces o<sup>f</sup> $\because \mathbb { R } ^ { d }$ . We d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> $\mathcal { P } _ { \mathrm { s u b } } ( d , k )$ over $( \mu , \Sigma )$ <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub> L<sub>e</sub>t $\mathcal { H } _ { d , k }$ b<sub>e</sub> th<sub>e un</sub>if<sub>orm</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on over</sub> � $\times k$

<sub>ma</sub>t<sub>r</sub>i<sub>ces w</sub>ith <sub>or</sub>th<sub>onorma</sub>l <sub>co</sub>l<sub>umns.</sub> Th<sub>en</sub>

$$
( \mu , \Sigma ) \sim \mathcal { P } _ { \mathrm { s u b } } ( d , k ) \iff \left\{ \begin{array} { l l } { \mu \sim { N } ( 0 , d ^ { 2 } \mathbb { I } _ { d } ) } \\ { \Sigma = D D ^ { \top } \ \mathrm { w h e r e } \ D \sim \mathcal { H } _ { d , k } . } \end{array} \right.\tag{5}
$$

This is equivalent to taking � to be the projector onto a uniformly-random (according to the Haar measure) �-dimensional subs<sub>p</sub>ace. We will omit the <sub>p</sub>arameters � and � from the notation when the<sub>y</sub> <sub>are</sub> <sub>c</sub>l<sub>ear</sub> f<sub>rom</sub> <sub>con</sub>t<sub>ex</sub>t<sub>.</sub>

Theorem 2.1 (Lower bound for private mean estimation with unknown covariance). There exists $c > 0$ such that for all $d , k , n \in \mathbb { N }$ with $k \leq$ �� and � log $n \leq c k .$ , every $( 1 , c / n ) – D P$ algorithm A satisfies:

$$
\operatorname* { P r } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { s u b } } \left[ \| \mathcal { R } ( X ) - \mu \| _ { 2 } \ge \frac { c \sqrt { d k } } { n } \right] \ge \frac { 1 } { 4 } .
$$

That is, for all $\varepsilon \leq 1$ and $\delta \leq { \frac { c } { n } }$ , we have $\begin{array} { r } { \mathfrak { R } _ { \varepsilon , \delta } { \left( \mathcal { P } _ { s u b } , n \right) } = \Omega { \left( \frac { \sqrt { d k } } { n } \right) } } \end{array}$

Recall that the al<sub>g</sub>orithm of [DJYZZ24] for the known covariance case obtains error

$$
\tilde { O } _ { \varepsilon , \delta } \left( \sqrt { \frac { \operatorname { t r } ( \Sigma ) } { n } } + \frac { \operatorname { t r } ( \Sigma ^ { 1 / 2 } ) } { n } \right) = \tilde { O } _ { \varepsilon , \delta } \left( \sqrt { \frac { k } { n } } + \frac { k } { n } \right) .
$$

C<sub>ompar</sub>i<sub>ng</sub> th<sub>e a</sub>b<sub>ove</sub> t<sub>o our</sub> l<sub>ower</sub> b<sub>oun</sub>d f<sub>or mean es</sub>ti<sub>ma</sub>ti<sub>on w</sub>ith <sub>un</sub>k<sub>nown covar</sub>i<sub>ance y</sub>i<sub>e</sub>ld<sub>s</sub> the followin<sub>g</sub> cost of not knowin<sub>g</sub> the covariance in <sub>p</sub>rivate mean estimation: In the settin<sub>g</sub> of Th<sub>eorem</sub> 2<sub>.</sub>1<sub>,</sub>

$$
\mathfrak { R } _ { \varepsilon , \delta } ( \mathcal { P } _ { \mathrm { s u b } } , n ) = \tilde { \Omega } \left( \sqrt { \frac { d } { k } } \right) \cdot \mathfrak { R } _ { \varepsilon , \delta } ^ { \mathrm { k n o w n } } ( \mathcal { P } _ { \mathrm { s u b } } , n ) .\tag{6}
$$

We note that this blowu<sub>p</sub> is ti<sub>g</sub>ht, since it matches the error of the folklore al<sub>g</sub>orithm (see Section B). T<sub>o</sub> b<sub>e</sub> <sub>prec</sub>i<sub>se,</sub> f<sub>or</sub> th<sub>e</sub> f<sub>am</sub>il<sub>y</sub> $\mathcal { P } _ { \mathrm { s u b } }$ , <sup>and</sup> p<sup>rivac</sup>y p<sup>arameters</sup> $\varepsilon = 1 , \delta = c / n$ <sub>,</sub> th<sub>e</sub> f<sub>o</sub>lkl<sub>ore a</sub>l<sub>gor</sub>ith<sub>m</sub> b<sub>ase</sub>d on [TCKMS22] and <sub>p</sub>resented in Section B achieves, with constant <sub>p</sub>robabilit<sub>y</sub>, error

$$
O \left( { \sqrt { \frac { k } { n } } } + { \frac { { \sqrt { d k } } ( \log n ) ^ { 3 / 2 } } { n } } \right)
$$

f<sub>or �</sub> l<sub>arger</sub> th<sub>an a un</sub>i<sub>versa</sub>l <sub>cons</sub>t<sub>an</sub>t<sub>.</sub> If $n \geq C k$ f<sub>or some cons</sub>t<sub>an</sub>t $C > 0 .$ <sub>,</sub> th<sub>en one can</sub> l<sub>earn</sub> th<sub>e</sub> k<sub>-</sub> dimensional subs<sub>p</sub>ace of Σ [SS21, Theorem 1.1, $\ell = k { - 1 } ]$ <sub>an</sub>d l<sub>earn</sub> th<sub>e mean w</sub>ith di<sub>mens</sub>i<sub>on-</sub>f<sub>ree error</sub> a<sup>f</sup>ter projecting t<sup>h</sup>e <sup>d</sup>ata on t<sup>h</sup>is su<sup>b</sup>space. Our <sup>l</sup>ower <sup>b</sup>oun<sup>d</sup> app<sup>l</sup>ies to t<sup>h</sup>e a<sup>l</sup>most-comp<sup>l</sup>ementary re<sub>g</sub>ime where <sub>�</sub> lo<sub>g</sub> $n \leq c k$ <sub>an</sub>d <sub>s</sub>t<sub>a</sub>t<sub>es</sub> th<sub>a</sub>t th<sub>e</sub> di<sub>mens</sub>i<sub>on-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>cos</sub>t <sub>o</sub>f <sub>pr</sub>i<sub>vacy</sub> <sub>o</sub>f th<sub>e</sub> f<sub>o</sub>lkl<sub>ore</sub> al<sub>g</sub>orithm is necessar<sub>y</sub> u<sub>p</sub> to <sub>p</sub>ol<sub>y</sub>lo<sub>g</sub>arithmic factors.

T<sub>o</sub> <sub>prove</sub> Th<sub>eorem</sub> 2<sub>.</sub>1<sub>,</sub> <sub>we</sub> fi<sub>rs</sub>t <sub>s</sub>h<sub>ow</sub> h<sub>ow</sub> t<sub>o</sub> t<sub>rans</sub>f<sub>orm</sub> <sub>any</sub> <sub>pr</sub>i<sub>va</sub>t<sub>e</sub> <sub>an</sub>d <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>accura</sub>t<sub>e</sub> <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> i<sub>n</sub>t<sub>o</sub> <sub>a</sub> <sub>pr</sub>i<sub>va</sub>t<sub>e</sub> “<sub>samp</sub>li<sub>ng</sub>” <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> th<sub>a</sub>t <sub>corre</sub>l<sub>a</sub>t<sub>es</sub> <sub>w</sub>ith th<sub>e</sub> <sub>popu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> m<sub>a</sub>trix<sub>.</sub>

Definition 2.2 $( ( \lambda , \alpha )$ -sampler). Fix $d , k , n \in \mathbb { N }$ <sub>an</sub>d $\alpha , \lambda > 0$ . An al<sub>g</sub>orithm M is a $( \lambda , \alpha )$ <sub>-samp</sub>l<sub>er</sub> f<sub>or</sub> $\mathcal { P } _ { \mathrm { s u b } }$ (defined in (5)) with sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> � if

$$
\begin{array} { r l r } {  { \operatorname* { P r } _ { \Sigma \sim \mathcal { P } _ { \mathrm { s u b } } } [ \mathcal { M } ( X ) ^ { \top } \Sigma \mathcal { M } ( X ) \ge \lambda ^ { 2 } ] \ge \frac { 1 } { 1 2 } \mathrm { ~ a n d ~ } \| \mathcal { M } ( X ) \| _ { 2 } \le \alpha . } } \\ & { } & { \quad \quad \mathrm { ~ } _ { X \sim \mathcal { N } ( 0 , \Sigma ) ^ { \otimes n } } } \\ & { } & { \quad \mathrm { ~ c o i n s ~ o f ~ } M } \end{array}
$$

Remark 1. Not all values of � and � are compatible with the existence of a $( \lambda , \alpha )$ -sam<sub>p</sub>ler. Triviall<sub>y,</sub> it f<sub>o</sub>ll<sub>ows</sub> di<sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> th<sub>a</sub>t <sub>necessar</sub>il<sub>y</sub> $\alpha \ge \lambda$ <sub>.</sub> I<sub>n</sub> f<sub>ac</sub>t<sub>, we w</sub>ill <sub>s</sub>h<sub>ow</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 2<sub>.</sub>2 th<sub>a</sub>t<sub>,</sub> f<sub>or</sub> $n \leq { \tilde { O } } ( k )$ <sub>,</sub> th<sub>e</sub> <sub>norm</sub> <sub>�</sub> <sub>mus</sub>t b<sub>e</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y</sub> l<sub>arger</sub> th<sub>an</sub> �<sub>.</sub>

O<sub>u</sub>r main technical contrib<sub>u</sub>tion is a lower bo<sub>u</sub>nd on the norm of an<sub>y</sub> <sub>p</sub>rivate $( \lambda , \alpha )$ <sub>-samp</sub>l<sub>er</sub> th<sub>a</sub>t <sub>uses</sub> f<sub>ew samp</sub>l<sub>es.</sub>

Theorem 2.3 (Lower bound for private sampling). There exists a constant $c > 0$ such that for all $d , k , n \in \mathbb { N }$ and $\lambda > 0$ $i f k \le$ �� and � log $n \leq c k$ then every $( \lambda , \alpha )$ -sampler for $\mathcal { P } _ { s u b }$ with sample complexity � that is $( 1 , c / n ) – D P$ requires $\alpha \geq c \lambda \sqrt { d / k }$

To <sub>p</sub>rove our main result<sub>,</sub> we show how to construct a <sub>p</sub>rivate sam<sub>p</sub>ler from a <sub>p</sub>rivate mean estimation al<sub>g</sub>orithm.

Lemma 2.4 (From mean estimation to sampling). There exists a constant $c > 0$ such that for all $d , k , n \in \mathbb { N }$ with $n \leq c k$ and $k \leq d$ the following holds: If A is a $( 1 , c / n ) – D P$ algorithm satisfying

$$
\mathop { \operatorname* { P r } } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { s u b } } [ \| \mathcal { A } ( X ) - \mu \| _ { 2 } \leq \alpha ] \geq \frac { 3 } { 4 } ,
$$

then there exists a $( c k / n , \alpha )$ -sampler for $\mathcal { P } _ { s u b }$ with sample complexity � that is $( 1 , c / n ) – D P .$

W<sub>e</sub> <sub>now</sub> <sub>com</sub>bi<sub>ne</sub> Th<sub>eorems</sub> 2<sub>.</sub>3 <sub>an</sub>d 2<sub>.</sub>4 t<sub>o</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f th<sub>e</sub> <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>.</sub>

Proof of Theorem 2.1. Let $c > 0$ be a suficientl<sub>y</sub> small constant, and let A be a $( 1 , c / n ) – \mathrm { D P }$ <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> achievin<sub>g</sub> the accurac<sub>y</sub> <sub>g</sub>uarantee in Theorem 2.4 with <sub>p</sub>arameter <sub>�</sub>. Then there exists a $( 1 , c / n ) – \mathrm { D P }$ mechanism M that is a $( c k / n , \alpha )$ <sub>-samp</sub>l<sub>er</sub> f<sub>or</sub> $\mathcal { P } _ { \mathrm { s u b } }$ <sub>w</sub>ith <sub>samp</sub>l<sub>e</sub> <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> <sub>�.</sub> It f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> Th<sub>eorem</sub> 2<sub>.</sub>3 th<sub>a</sub>t f<sub>or �</sub> l<sub>og</sub> $n < c k$ <sub>, we mus</sub>t h<sub>ave</sub> $\alpha \geq \Omega ( \sqrt { d k } / n )$ <sub>, w</sub>hi<sub>c</sub>h <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e resu</sub>lt<sub>.</sub> □

The remainder of the section is or<sub>g</sub>anized as follows: In Section 2.1 we <sub>p</sub>rove Theorem 2.4 (the reduction from <sub>p</sub>rivate mean estimation to <sub>p</sub>rivate sam<sub>p</sub>lin<sub>g</sub>), and in Section 2.2 we <sub>p</sub>rove Theorem 2.3 (the lower bound on the norm of the sam<sub>p</sub>ler).

## 2.1 Reduction from sampling

In this section<sub>,</sub> we construct a sam<sub>p</sub>ler that uses a mean estimation al<sub>g</sub>orithm as a subroutine. The <sub>samp</sub>l<sub>er use</sub>d i<sub>n our</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>cons</sub>i<sub>s</sub>t<sub>s essen</sub>ti<sub>a</sub>ll<sub>y o</sub>f th<sub>e c</sub>li<sub>ppe</sub>d <sub>error o</sub>f th<sub>e mean es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm.</sub>

Proof of Theorem 2.4. For a vector $y \in \mathbb { R } ^ { d }$ <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> $\mathrm { p r o j } _ { \alpha } ( y )$ as t<sup>h</sup>e projection o<sup>f</sup> � into t<sup>h</sup>e $\ell _ { 2 }$ b<sub>a</sub>ll <sub>o</sub>f <sub>ra</sub>di<sub>us</sub> <sub>�</sub> <sub>cen</sub>t<sub>ere</sub>d <sub>a</sub>t 0<sub>.</sub> F<sub>or</sub> $\rho = d _ { \mathrm { { \small i } } }$ , let M denote the followin<sub>g</sub> al<sub>g</sub>orithm:

$$
{ \mathcal { M } } ( X ) : { \mathrm { S a m p l e ~ } } \mu \sim { \mathcal { N } } { \bigl ( } 0 , \rho ^ { 2 } \mathbb { I } _ { d } { \bigr ) } . { \mathrm { R e t u r n ~ p r o j } } _ { \alpha } ( { \mathcal { A } } ( X + \mu ) - \mu ) .\tag{7}
$$

Since A is $( \varepsilon , \delta ) \mathrm { - D P }$ , <sub>p</sub>ost<sub>p</sub>rocessin<sub>g</sub> immediatel<sub>y</sub> im<sub>p</sub>lies that M is $( \varepsilon , \delta ) \mathrm { - D P }$ <sub>.</sub> It r<sub>e</sub>m<sub>a</sub>in<sub>s</sub> t<sub>o s</sub>h<sub>ow</sub> that M is a $( \lambda , \alpha )$ -sam<sub>p</sub>ler for a<sub>pp</sub>ro<sub>p</sub>riate �. Recall that for ever<sub>y</sub> Σ we denote $\lVert \cdot \rVert _ { \Sigma }$ th<sub>e</sub> <sub>pseu</sub>d<sub>o-</sub> <sub>norm</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> $\Sigma , \mathrm { e . g . } \parallel M ( X ) \parallel _ { \Sigma } = \sqrt { M ( X ) ^ { \top } \Sigma M ( X ) }$ . Let $G _ { 1 } : = \{ \| \mathcal { A } ( X + \mu ) - \mu \| _ { 2 } \leq \alpha \}$ b<sub>e</sub> th<sub>e</sub> <sub>even</sub>t <sub>un</sub>d<sub>er</sub> <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>c</sub>li<sub>pp</sub>i<sub>ng</sub> d<sub>oes</sub> <sub>no</sub>t <sub>mo</sub>dif<sub>y</sub> <sub>any</sub>thi<sub>ng,</sub> <sub>an</sub>d f<sub>or</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>sma</sub>ll $c > 0$ <sub>,</sub> l<sub>e</sub>t $G _ { 2 } : = \left\{ \| \mathcal { A } ( X + \mu ) - \mu \| _ { \Sigma } \geq \frac { c \hat { k } } { n } \right\}$ b<sub>e</sub> th<sub>e</sub> <sub>even</sub>t th<sub>a</sub>t th<sub>e</sub> <sub>raw</sub> <sub>ou</sub>t<sub>pu</sub>t i<sub>s</sub> l<sub>arge.</sub> Th<sub>en,</sub> <sub>con</sub>diti<sub>one</sub>d <sub>on</sub> $G _ { 1 }$ <sub>an</sub>d $G _ { 2 }$ <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t

$$
\left\| \operatorname { p r o j } _ { \alpha } ( { \mathcal { A } } ( X + \mu ) - \mu ) \right\| _ { \Sigma } = \left\| { \mathcal { A } } ( X + \mu ) - \mu \right\| _ { \Sigma } \geq { \frac { c k } { n } } .
$$

W<sub>e</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t b<sub>o</sub>th <sub>even</sub>t<sub>s</sub> h<sub>o</sub>ld <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $1 / 1 2$ . Recall that b<sub>y</sub> the <sub>g</sub>uarantee of A we have $\operatorname* { P r } [ \| { \mathcal { A } } ( X + \mu ) - \mu \| _ { 2 } \leq \alpha ] \geq 3 / 4$ <sub>, an</sub>d th<sub>us</sub> $\operatorname* { P r } [ G _ { 1 } ] \geq 3 / 4$ <sub>.</sub> N<sub>ex</sub>t <sub>we s</sub>h<sub>ow</sub> th<sub>a</sub>t $\mathrm { P r } [ G _ { 2 } ] \ge 1 / 3$ <sub>.</sub> O<sub>u</sub>r <sub>proo</sub>f <sub>uses</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> f<sub>o</sub>lkl<sub>ore</sub> <sub>resu</sub>lt<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>a</sub> B<sub>ayes</sub>i<sub>an</sub> <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f <sub>we</sub>ll<sub>-</sub>k<sub>nown</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> on <sub>p</sub>rivate mean estimation under a Gaussian distribution via fin<sub>g</sub>er<sub>p</sub>rintin<sub>g</sub> ar<sub>g</sub>uments [KLSU19], the s<sub>p</sub>ecific statement we use is from [GLSUZ26].

Proposition 2.5 (Folklore Result). Let $k , n \in \mathbb { N }$ . There exist absolute constants $c , C > 0$ , such that for any $( \varepsilon , \delta )$ -mechanism $\mathcal { A } : \mathbb { R } ^ { k \times n } \longrightarrow \mathbb { R } ^ { k }$ with $\begin{array} { r } { \delta \le \frac { c } { n } , a n y \varepsilon \le 1 } \end{array}$ , and any $\rho \geq C \operatorname* { m a x } \left( 1 , { \sqrt { k } } / n \right)$

$$
\operatorname* { P r } _ { \mu \sim N ( 0 , \rho ^ { 2 } \mathbb { I } _ { k } ) } \left[ \| \mathcal { A } ( X ) - \mu \| _ { 2 } \ge \frac { c k } { n } \right] \ge \frac { 1 } { 3 } .
$$

Fix $\Sigma = D D ^ { \top }$ <sub>.</sub> W<sub>e</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>ere</sub> <sub>ex</sub>i<sub>s</sub>t<sub>s</sub> <sub>an</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> $\mathcal { A } _ { \Sigma }$ <sub>suc</sub>h th<sub>a</sub>t if $\mu _ { k } \sim \mathcal { N } ( 0 , \rho ^ { 2 } \cdot \mathbb { I } _ { k } )$ <sub>an</sub>d $Y \sim \mathcal { N } ( \mu _ { k } , \mathbb { I } _ { k } ) ^ { \otimes n }$ th<sub>en</sub> $\| \mathcal { A } _ { \Sigma } ( { \pmb Y } ) - { \pmb \mu } _ { k } \| _ { 2 } = \| \mathcal { A } ( { \pmb X } ) - { \pmb \mu } \| _ { \Sigma }$ <sub>, w</sub>h<sub>ere</sub> $X \ \sim \ { \cal N } ( \mu , \Sigma ) ^ { \otimes n }$ <sub>an</sub>d $\mu \sim { \cal N } ( 0 , \rho ^ { 2 } \cdot \mathbb { I } _ { d } )$ <sub>.</sub> Thi<sub>s</sub> <sub>w</sub>ill <sub>a</sub>ll<sub>ow</sub> <sub>us</sub> t<sub>o</sub> <sub>use</sub> Th<sub>eorem</sub> 2<sub>.</sub>5 t<sub>o</sub> b<sub>oun</sub>d th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>even</sub>t $G _ { 2 }$ . Let $\mathcal { A } _ { \Sigma }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng a</sub>l<sub>gor</sub>ith<sub>m,</sub>

$$
\mathcal { A } _ { \Sigma } ( Y ) : \mathrm { ~ S a m p l e ~ } \mu _ { \bot } \sim \mathcal { N } ( 0 , \rho ^ { 2 } \cdot ( \mathbb { I } _ { d } - \Sigma ) ) . \mathrm { ~ S e t ~ } x _ { i } = D y _ { i } + \mu _ { \bot } \mathrm { . ~ R e t u r n ~ } D ^ { \top } \mathcal { R } ( X ) .\tag{8}
$$

U<sub>s</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $D ^ { \top } D = \mathbb { I } _ { k }$ <sub>an</sub>d $D ^ { \top } \pmb { \mu } _ { \bot } = 0$ <sub>we see</sub> th<sub>a</sub>t

$$
\begin{array} { r l } { \left\| D ^ { \top } \mathcal { A } ( X ) - \pmb { \mu } _ { k } \right\| _ { 2 } = \left\| D ^ { \top } \mathcal { A } ( X ) - D ^ { \top } D \pmb { \mu } _ { k } \right\| _ { 2 } } & { } \\ { = \left\| D ^ { \top } ( \mathcal { A } ( X ) - D \pmb { \mu } _ { k } - \pmb { \mu } _ { \perp } ) \right\| _ { 2 } } & { } \\ { = \| \mathcal { A } ( X ) - \pmb { \mu } \| _ { \Sigma } } \end{array}
$$

B<sub>y</sub> the distribution of $\mu _ { k }$ <sub>an</sub>d $\mu _ { \perp }$ <sub>,</sub> th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f $\mu = D \mu _ { k } + \mu _ { \perp }$ i<sub>s</sub> ${ \cal N } ( 0 , \rho ^ { 2 } \cdot \mathbb { I } _ { d } )$ . Moreover<sub>,</sub> b<sub>y</sub> th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f $\mathbf { \nabla } _ { \mathbf { y } _ { i : } }$ <sub>, we</sub> h<sub>ave, con</sub>diti<sub>one</sub>d <sub>on</sub> $\mu ,$ th<sub>a</sub>t $x _ { i } \sim { \mathcal { N } } ( \mu , \Sigma )$ <sub>.</sub> Th<sub>us,</sub> th<sub>e even</sub>t $G _ { 2 }$ d<sub>e</sub>fi<sub>ne</sub>d <sub>a</sub>b<sub>ove</sub> i<sub>s equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> th<sub>e even</sub>t th<sub>a</sub>t $\begin{array} { r } { \| \mathcal { A } _ { \Sigma } ( Y ) - \mu _ { k } \| _ { 2 } \geq \frac { c k } { n } } \end{array}$ <sub>,</sub> which<sub>,</sub> invokin<sub>g</sub> Theorem 2.5 as $\rho = d \ge k \ge c ^ { - 1 / 2 } \operatorname* { m a x } ( 1 , \sqrt { k } / n )$ <sub>,</sub> <sub>occurs</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $\frac 1 3$

A <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>over</sub> $G _ { 1 }$ <sub>an</sub>d $G _ { 2 }$ <sub>now</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>a</sub>t $\operatorname* { P r } [ \| \mathcal { M } ( X ) \| _ { \Sigma } \geq c k / n ] \geq 1 / 1 2$ <sub>.</sub> M<sub>oreover,</sub> th<sub>e</sub> definition of M enforces that $\| M ( X ) \| _ { 2 } \leq \alpha .$ , and thus M is a $( c k / n , \alpha ) – \mathrm { s a m p l e r } .$ □

## 2.2 Private sampling lower bound

W<sub>e co</sub>n<sub>s</sub>id<sub>e</sub>r <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> $\Sigma \sim \mathcal { P } _ { \mathrm { s u b } }$ <sub>,</sub> d<sub>a</sub>t<sub>ase</sub>t $\pmb { X } = \pmb { x } _ { 1 } , . . . , \pmb { x } _ { n } \sim \pmb { N } ( 0 , \pmb { \Sigma } ) ^ { \otimes n }$ , <sup>out</sup>p<sup>ut</sup> $\pmb { y } = \mathcal { M } ( \pmb { X } )$ <sub>,</sub> test <sub>p</sub>o<sup>i</sup>nt $\boldsymbol { z } \sim \mathcal { N } ( \boldsymbol { 0 } , \boldsymbol { \Sigma } )$ <sub>.</sub> T<sub>o</sub> <sub>prove</sub> <sub>a</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> <sub>norm</sub> <sub>o</sub>f th<sub>e</sub> <sub>samp</sub>l<sub>er,</sub> <sub>we</sub> <sub>cons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> test statistic. For fixed Σ and $y , z \in \mathbb { R } ^ { d }$ <sub>w</sub>ith $y ^ { \top } \Sigma y > 0$ l<sub>e</sub>t

$$
T _ { \Sigma , y } ( z ) = \frac { \langle y , z \rangle ^ { 2 } } { y ^ { \top } \Sigma y } .
$$

If $y ^ { \top } \Sigma y = 0$ <sub>,</sub> th<sub>en we se</sub>t $T _ { \Sigma , y } ( z ) = 0$ <sub>.</sub> W<sub>e</sub> <sub>w</sub>ill <sub>om</sub>it th<sub>e</sub> d<sub>epen</sub>d<sub>ence</sub> <sub>on</sub> $\Sigma$ <sub>an</sub>d <sub>�</sub> <sub>w</sub>h<sub>en</sub> th<sub>ey</sub> <sub>are</sub> <sub>c</sub>l<sub>ear</sub> f<sub>rom con</sub>t<sub>ex</sub>t <sub>an</sub>d i<sub>ns</sub>t<sub>ea</sub>d <sub>wr</sub>it<sub>e</sub> $T ( z )$ <sub>.</sub> Th<sub>eorem</sub> 2<sub>.</sub>3 f<sub>o</sub>ll<sub>ows</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>wo</sub> L<sub>emma</sub>t<sub>a:</sub> O<sub>ur</sub> fi<sub>rs</sub>t <sub>resu</sub>lt <sub>prov</sub>id<sub>es</sub> <sub>a</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> t<sub>es</sub>t i<sub>n</sub> th<sub>e</sub> OUT <sub>case</sub> th<sub>a</sub>t h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> <sub>a</sub>ll fi<sub>xe</sub>d <sub>�</sub> <sub>an</sub>d Σ with hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub> over the test <sub>p</sub>oint z.

Lemma 2.6 (OUT case upper bound). For every $c \in ( 0 , 1 )$ there exists $C > 0$ such that for all $\boldsymbol { y } \in \mathbb { R } ^ { d }$ and $\Sigma \succeq 0 , i f k \geq C n$ log � then

$$
\operatorname* { P r } _ { z \sim N ( 0 , \Sigma ) } \left[ T _ { \Sigma , y } ( z ) \geq \frac { k } { 4 n } \right] \leq \frac { c } { n } .
$$

ProofofTheorem 2.6. The proof follows from the distribution of $T _ { \Sigma , y } ( z )$ for fixed <sub>�</sub> and Σ. Observe th<sub>a</sub>t $\langle y , z \rangle \sim N ( 0 , y ^ { \top } \Sigma y )$ <sub>, an</sub>d th<sub>us</sub> $\langle y , z \rangle / \sqrt { y ^ { \top } \Sigma y } \sim N ( 0 , 1 )$ <sub>.</sub> Sin<sub>ce</sub> $T _ { \Sigma , y } ( z ) = ( \langle y , z \rangle / \sqrt { y ^ { \top } \Sigma y } ) ^ { 2 }$ we <sub>see</sub> th<sub>a</sub>t $T _ { \Sigma , y } ( z ) \sim \chi _ { 1 } ^ { 2 } .$ Standard concentration bounds for chi-s<sub>q</sub>uare random variables (Theorem A.5) <sub>y</sub>ield the result. □

N<sub>ex</sub>t<sub>,</sub> <sub>we</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub> <sub>ana</sub>l<sub>ogous</sub> <sub>resu</sub>lt f<sub>or</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f th<sub>e</sub> t<sub>es</sub>t i<sub>n</sub> th<sub>e</sub> IN <sub>case.</sub> W<sub>e</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>over</sub> th<sub>e</sub> randomness of the data X, the mechanism’s out<sub>p</sub>ut $\pmb { y } = \pmb { M } ( \pmb { X } )$ <sub>,</sub> th<sub>e covar</sub>i<sub>ance</sub> $\Sigma ,$ <sub>an</sub>d th<sub>e c</sub>h<sub>o</sub>i<sub>ce o</sub>f index i, $T ( x _ { i } )$ i<sub>s</sub> “l<sub>arge</sub>” <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $\Omega ( 1 / n )$

Lemma 2.7 (IN case lower bound). Let $c > 0$ be a suficiently small constant. Suppose $n \leq c k$ and $\alpha \leq c \lambda \sqrt { d / k }$ . Then, $i f i \sim [ n ]$ is chosen uniformly

$$
\operatorname* { P r } _ { X , y , \Sigma , i } \left[ T _ { \Sigma , y } ( x _ { i } ) \geq { \frac { k } { 4 n } } \right] \geq { \frac { c } { n } } .
$$

Proof of Theorem 2.7. Throughout the proof, we let $C > 0$ d<sub>eno</sub>t<sub>e</sub> <sub>a</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge</sub> <sub>cons</sub>t<sub>an</sub>t <sub>an</sub>d $c > 0$ <sub>a sma</sub>ll <sub>cons</sub>t<sub>an</sub>t<sub>.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>ere ex</sub>i<sub>s</sub>t<sub>s an even</sub>t � <sub>over</sub> th<sub>e samp</sub>li<sub>ng o</sub>f $X , y$ <sub>an</sub>d $\Sigma$ <sub>suc</sub>h th<sub>a</sub>t $\operatorname* { P r } [ G ] \geq c _ { \mathrm { i } }$ <sub>,</sub> <sub>an</sub>d f<sub>or</sub> <sub>a</sub>ll $( X , y , \Sigma )$ in $G ,$ th<sub>e</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f $T ( x _ { i } )$ over i is at least $4 k / n$ Th<sub>us,</sub> th<sub>ere</sub> i<sub>s some</sub> � <sub>suc</sub>h th<sub>a</sub>t $T ( \mathbf { \boldsymbol { x } } _ { i } ) \geq 4 k / n _ { \mathrm { \Omega } }$ , which, since i is uniform in [�] will sufice to <sub>p</sub>rove th<sub>e</sub> l<sub>emma.</sub> W<sub>e</sub> b<sub>eg</sub>i<sub>n</sub> b<sub>y</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> th<sub>e even</sub>t �<sub>.</sub>

Let $P _ { X }$ <sup>b</sup>e t<sup>h</sup>e ort<sup>h</sup>ogona<sup>l</sup> projection onto t<sup>h</sup>e span o<sup>f</sup> $X ,$ <sub>,</sub> <sub>an</sub>d l<sub>e</sub>t $\hat { \Sigma } _ { X } = { \scriptstyle { \frac { 1 } { n } } } X X ^ { \top }$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> $G =$ $G _ { 1 } \cap G _ { 2 } \cap G _ { 3 }$ <sub>as</sub> th<sub>e even</sub>t th<sub>a</sub>t th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>ree con</sub>diti<sub>ons</sub> h<sub>o</sub>ld<sub>:</sub>

$$
\begin{array} { r } { G _ { 1 } \colon \frac { k } { 2 n } \cdot P _ { X } \preceq \hat { \Sigma } _ { X } \preceq \frac { 2 k } { n } \cdot P _ { X } . } \end{array}
$$

$$
\begin{array} { r } { G _ { 2 } \colon y ^ { \top } ( \Sigma - P _ { X } ) y \leq \frac { 2 C \alpha ^ { 2 } k } { d } . } \end{array}
$$

$$
\begin{array} { r } { G _ { 3 } \colon y ^ { \top } \Sigma y \geq \lambda ^ { 2 } . } \end{array}
$$

Intuitivel<sub>y</sub>, the event � occurs if � is a t<sub>yp</sub>ical sam<sub>p</sub>le, if the norm of <sub>�</sub> in the S<sub>p</sub>an(Σ) \S<sub>p</sub>an(�) is at most a $\frac { k } { d }$ fraction of the total norm of <sub>�</sub>, and if the norm of <sub>�</sub> in S<sub>p</sub>an(Σ) is at least the re<sub>q</sub>uired norm $\lambda ^ { 2 }$ . We first show that the ex<sub>p</sub>ectation over i is lar<sub>g</sub>e for ever<sub>y</sub> $X , y$ and Σ in �.

Claim 2.8. Suppose that $\begin{array} { r } { \alpha \leq \lambda \sqrt { \frac { d } { 4 C k } } } \end{array}$ and that �, �, and Σ are in �. Then $\begin{array} { r } { \mathbb { E } _ { i } \left[ T _ { \Sigma , y } ( x _ { i } ) \right] \geq \frac { k } { 4 n } } \end{array}$

Proof. The essence of the proof is the observation that

$$
\operatorname { \mathbb { E } } _ { i } [ T ( x _ { i } ) ] = \frac { y ^ { \top } \operatorname { \mathbb { E } } [ x _ { i } x _ { i } ^ { \top } ] y } { y ^ { \top } \Sigma y } = \frac { y ^ { \top } \hat { \Sigma } _ { X } y } { y ^ { \top } \Sigma y } .
$$

The remainder of the <sub>p</sub>roof follows b<sub>y</sub> al<sub>g</sub>ebraic mani<sub>p</sub>ulation<sub>,</sub> and b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the ine<sub>q</sub>ualities th<sub>a</sub>t h<sub>o</sub>ld f<sub>or a</sub>ll $X , y .$ , and Σ in �.

$$
\frac { y ^ { \top } \hat { \Sigma } _ { X } y } { y ^ { \top } \Sigma y } \geq \frac { k } { 2 n } \cdot \frac { y ^ { \top } P _ { X } y } { y ^ { \top } \Sigma y }\tag{�<sub>1</sub>}
$$

$$
\geq \frac { k } { 2 n } \cdot \frac { y ^ { \top } \Sigma y - 2 C \alpha ^ { 2 } k / d } { y ^ { \top } \Sigma y }\tag{�<sub>2</sub>}
$$

$$
= { \frac { k } { 2 n } } \left( 1 - { \frac { 2 C \alpha ^ { 2 } k } { d y ^ { \top } \Sigma y } } \right) \geq { \frac { k } { 2 n } } \left( 1 - { \frac { 2 C \alpha ^ { 2 } k } { d \lambda ^ { 2 } } } \right)\tag{�<sub>3</sub>}
$$

$$
\geq { \frac { k } { 4 n } } .
$$

O<sub>u</sub>r second claim states that � occ<sub>u</sub>rs with constant <sub>p</sub>robabilit<sub>y</sub>.

Claim 2.9. There exists a constant $c > 0$ such that for all suficiently large $k \leq$ �� and $n \leq c k$ , we have Pr ${ \cdot } _ { X , y , \Sigma } [ G ] \geq c$

Proof. Let $C > 0$ b<sub>e a su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge cons</sub>t<sub>an</sub>t<sub>.</sub> A<sub>n</sub>d <sub>reca</sub>ll th<sub>a</sub>t $G _ { 1 } , G _ { 2 }$ <sub>an</sub>d $G _ { 3 }$ <sub>are</sub> th<sub>e even</sub>t<sub>s</sub> com<sub>p</sub>osin<sub>g</sub> �. We first bound $G _ { 1 }$ <sub>an</sub>d $G _ { 3 } { \mathrm { : } }$ : B<sub>y</sub> standard results in random matrix theor<sub>y</sub> (see Th<sub>eorem</sub> $\mathrm { A } . 1 )$ <sub>, as</sub> l<sub>ong as</sub> � i<sub>s a</sub>t l<sub>eas</sub>t <sub>some su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge cons</sub>t<sub>an</sub>t<sub>,</sub> $\operatorname* { P r } [ G _ { 1 } ] \ \geq \ 1 - \ 1 / C$ Additi<sub>ona</sub>ll<sub>y,</sub> b<sub>y</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f <sub>an</sub> $( \lambda , \alpha )$ -sam<sub>p</sub>ler (Theorem 2.2) we have $\begin{array} { r } { \operatorname* { P r } [ G _ { 3 } ] \geq \frac { 1 } { 1 2 } } \end{array}$

It <sub>rema</sub>i<sub>ns</sub> t<sub>o</sub> b<sub>oun</sub>d th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f $G _ { 2 }$ . The ke<sub>y</sub> idea is that since � is <sub>u</sub>niforml<sub>y</sub> distrib<sub>u</sub>ted on t<sup>h</sup>e space o<sup>f</sup> ran<sup>k</sup> <sup>�</sup> ort<sup>h</sup>ogona<sup>l</sup> projections in $\mathbb { R } ^ { i }$ , conditioned on X, the matrix $\Sigma - P _ { X }$ i<sub>s</sub> <sub>a</sub> <sub>ran</sub>k $k - n$ ran<sup>d</sup>om ort<sup>h</sup>ogona<sup>l</sup> projection wit<sup>h</sup>in t<sup>h</sup>e <sup>l</sup>inear su<sup>b</sup>space $\operatorname { S p a n } ( P _ { X } ) ^ { \perp }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>a</sub> $d - n$ dimensional s<sub>p</sub>ace. Th<sub>u</sub>s<sub>,</sub>

$$
\operatorname { \mathbb { E } } [ \Sigma - P _ { X } \mid X ] = { \frac { k - n } { d - n } } ( \mathbb { I } _ { d } - P _ { X } ) \preceq { \frac { 2 k } { d } } \mathbb { I } _ { d } .
$$

Sin<sub>ce</sub> $\| y \| _ { 2 } \leq \alpha$ <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t $\mathbb { E } [ y ^ { \top } ( \Sigma - P _ { X } ) y ] \le 2 \alpha ^ { 2 } k / d$ <sub>,</sub> <sub>an</sub>d h<sub>ence,</sub> b<sub>y</sub> M<sub>ar</sub>k<sub>ov</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y,</sub>

$$
\operatorname* { P r } [ G _ { 2 } ] \ge 1 - \operatorname* { P r } \left[ y ^ { \top } ( \Sigma - P _ { X } ) y \ge C \mathbb { E } [ y ^ { \top } ( \Sigma - P _ { X } ) y ] \right] \ge 1 - \frac { 1 } { C } .
$$

T<sub>o</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m,</sub> <sub>o</sub>b<sub>serve</sub> th<sub>a</sub>t th<sub>e</sub> <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> $\mathrm { P r } [ G _ { 1 } \cap G _ { 2 } \cap G _ { 3 } ] \geq$ $\textstyle { \frac { 1 } { 1 2 } } - { \frac { 2 } { C } } \geq c$ f<sub>or</sub> $C \geq 4 8$ <sub>an</sub>d $c \leq 1 / 2 4$ □

We no<sub>w</sub> com<sub>p</sub>lete the <sub>p</sub>roof of Theorem 2.7. Since for e<sub>v</sub>er<sub>y</sub> $X , y , \Sigma$ in $G _ { s }$ th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $i ^ { * }$ <sub>suc</sub>h th<sub>a</sub>t $T ( \pmb { x } _ { i ^ { * } } ) \geq \mathbb { E } _ { i } [ T ( \pmb { x } _ { i } ) ]$ <sub>, we can conc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>a</sub>t

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } _ { X , y , \Sigma , i } \left[ T ( { \boldsymbol x } _ { i } ) \geq \frac { k } { 4 n } \right] \geq \operatorname* { P r } \left[ T ( { \boldsymbol x } _ { i } ) \geq \frac { k } { 4 n } \mid G \right] \operatorname* { P r } [ G ] } \\ { \displaystyle \qquad \geq \operatorname* { P r } \left[ T ( { \boldsymbol x } _ { i } ) \geq \operatorname* { \mathbb { E } } _ { i } \left[ T ( { \boldsymbol x } _ { i } ) \mid X , y , \Sigma \right] \mid G \right] \operatorname* { P r } [ G ] } \\ { \displaystyle \qquad \geq \operatorname* { P r } [ i = i ^ { * } ] \operatorname* { P r } [ G ] } \\ { \displaystyle \qquad \geq \frac { c } { n } . } \end{array}
$$

Th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> Th<sub>eorem</sub> 2<sub>.</sub>8 <sub>an</sub>d th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> Th<sub>eorem</sub> 2<sub>.</sub>9 and the fact that i is uniform in [�] and inde<sub>p</sub>endent of $X , y$ <sub>an</sub>d $\Sigma .$ □

We concl<sub>u</sub>de this section b<sub>y</sub> combinin<sub>g</sub> Theorems 2.6 and 2.7 to <sub>p</sub>ro<sub>v</sub>e Theorem $2 . 3 ,$ <sub>w</sub>hi<sub>c</sub>h <sub>s</sub>t<sub>a</sub>t<sub>es</sub> t<sup>h</sup>at ever<sub>y</sub> $( \lambda , \alpha )$ <sub>-samp</sub>l<sub>er</sub> th<sub>a</sub>t i<sub>s</sub> $( 1 , c / n ) – \mathrm { D P }$ <sub>an</sub>d <sub>uses</sub> $n \leq k / \log n$ <sub>samp</sub>l<sub>es</sub> h<sub>as</sub> <sub>norm</sub> $\alpha \geq c \lambda { \sqrt { d / k } }$ . It <sub>w</sub>ill b<sub>e</sub> <sub>conven</sub>i<sub>en</sub>t t<sub>o</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>no</sub>t<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> i<sub>n</sub>di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>a</sub>bilit<sub>y.</sub>

Definition 2.10 $( ( \varepsilon , \delta )$ -indistinguishability). Two distributions $\mathcal { P } , \boldsymbol { Q }$ <sub>over a</sub> d<sub>oma</sub>i<sub>n</sub> $\mathcal { W }$ are $( \varepsilon , \delta ) \cdot$ i<sub>n</sub>di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>a</sub>bl<sub>e,</sub> d<sub>eno</sub>t<sub>e</sub>d $\mathcal { P } \approx _ { \varepsilon , \delta } Q , \mathrm { i f } ,$ f<sub>or any measura</sub>bl<sub>e su</sub>b<sub>se</sub>t $W \subset \mathcal { W }$

$$
{ \mathcal P } ( W ) \le e ^ { \varepsilon } Q ( W ) + \delta \mathrm { ~ a n d ~ } Q ( W ) \le e ^ { \varepsilon } { \mathcal P } ( W ) + \delta .
$$

If $_ { \pmb { x } }$ <sub>an</sub>d $\mathbf { x } ^ { \prime }$ <sub>are ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>es w</sub>ith di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\mathcal { P }$ <sub>an</sub>d $\boldsymbol { Q }$ res<sub>p</sub>ectivel<sub>y</sub>, we write x $\approx _ { \varepsilon , \delta }$ x<sup>′</sup> to i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> th<sub>a</sub>t $\mathcal { P } \approx _ { \varepsilon , \delta } Q$

Proof of Theorem 2.3. The proof follows by contradiction. Suppose M is an $( \varepsilon , \delta ) \ – \mathrm { D P } \left( \lambda , \alpha \right)$ -sam<sub>p</sub><sup>l</sup>er <sub>w</sub>ith $\varepsilon \le 1 , \delta \le c / 2 n$ <sub>an</sub>d $\alpha \leq c \lambda \sqrt { d / k }$ <sub>.</sub> F<sub>o</sub>r <sub>eve</sub>r<sub>y</sub> ind<sub>e</sub>x $i \in [ n ]$ l<sub>e</sub>t $\pmb { X } ^ { \oplus i }$ denote X with $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } _ { i } }$ <sub>rep</sub>l<sub>ace</sub>d b<sub>y</sub> z. Then, since M is $( \varepsilon , \delta ) \mathrm { - D P }$ <sub>we</sub> h<sub>ave</sub>

$$
T _ { \mathcal M ( X ) } ( z ) \approx _ { \varepsilon , \delta } T _ { \mathcal M ( X ^ { \oplus i } ) } ( z ) \equiv T _ { \mathcal M ( X ) } ( x _ { i } )
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>equa</sub>lit<sub>y</sub> i<sub>n</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ and z are both sam<sub>p</sub>led from $N ( 0 , \Sigma )$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y o</sub>f th<sub>e o</sub>th<sub>er</sub> d<sub>a</sub>t<sub>apo</sub>i<sub>n</sub>t<sub>s.</sub> H<sub>ence,</sub> if $y \ = \ M ( X )$ <sub>an</sub>d $\textit { i } \sim [ n ]$ <sub>,</sub> th<sub>en</sub> b<sub>y</sub> Th<sub>eorems</sub> 2<sub>.</sub>6 <sub>an</sub>d 2<sub>.</sub>7 th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a cons</sub>t<sub>an</sub>t $C > 0$ (de<sub>p</sub>endin<sub>g</sub> onl<sub>y</sub> on �) such that if $k \geq C n$ <sup>l</sup>o<sub>g</sub> �

$$
\frac { c } { n } \leq \operatorname* { P r } \left[ T _ { \mathcal { M } ( X ) } ( x _ { i } ) \geq \frac { k } { 4 n } \right] \leq e ^ { \varepsilon } \operatorname* { P r } \left[ T _ { \mathcal { M } ( X ) } ( z ) \geq \frac { k } { 4 n } \right] + \delta < \frac { c } { 2 n } + \frac { c } { 2 n } ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s a con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on.</sub> Th<sub>ere</sub>f<sub>ore, every suc</sub>h <sub>samp</sub>l<sub>er mus</sub>t h<sub>ave</sub> $\alpha \geq c \lambda { \sqrt { d / k } }$

## 3 Lower bound for mean estimation with diagonal covariance

In this section <sub>w</sub>e sho<sub>w</sub> a lo<sub>w</sub>er bo<sub>u</sub>nd for <sub>p</sub>ri<sub>v</sub>ate mean estimation <sub>w</sub>hen the co<sub>v</sub>ariance matrix is <sub>prom</sub>i<sub>se</sub>d t<sub>o</sub> b<sub>e</sub> di<sub>agona</sub>l<sub>.</sub> O<sub>ur</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>uses</sub> th<sub>e</sub> <sub>sca</sub>l<sub>e</sub>d<sub>-</sub>i<sub>nverse-c</sub>hi<sub>-square</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>as</sub> <sub>a</sub> <sub>pr</sub>i<sub>or</sub> <sub>over</sub> th<sub>e var</sub>i<sub>ances</sub> i<sub>n eac</sub>h <sub>coor</sub>di<sub>na</sub>t<sub>e.</sub>

Definition 3.1 $( \mathsf { i n v } \chi ^ { 2 } ( \tau , \nu ) )$ . The scaled-inverse-chi-squared distribution with � degrees of freedom <sub>an</sub>d <sub>sca</sub>l<sub>e</sub> <sub>�</sub> i<sub>s</sub> th<sub>e</sub> l<sub>aw</sub> <sub>w</sub>ith d<sub>ens</sub>it<sub>y,</sub> f<sub>or</sub> $x \ge 0$ , <sub>g</sub><sup>i</sup>ven <sup>b</sup><sub>y</sub>

$$
f _ { \tau , \nu } ( x ) \propto \frac { \exp \left[ \frac { - \nu \tau ^ { 2 } } { 2 x } \right] } { x ^ { 1 + \nu / 2 } } .
$$

E<sub>q</sub>uivalentl<sub>y</sub> $\sigma ^ { 2 } \sim \mathsf { i n v } \chi ^ { 2 } ( \nu , \tau ^ { 2 } )$ <sub>,</sub> if ${ \frac { \nu \tau ^ { 2 } } { \sigma ^ { 2 } } } \sim \chi _ { \nu } ^ { 2 } .$

W<sub>e</sub> th<sub>en</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> $\mathcal { P } _ { \mathrm { d i a g } }$ over instance $( \mu , \Sigma )$ <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

$$
( \mu , \Sigma ) \sim \mathcal { P } _ { \mathrm { d i a g } } ( d ) \iff \left\{ \begin{array} { l l } { \mu \sim \mathcal { N } ( 0 , d ^ { 2 } \mathbb { I } _ { d } ) } \\ { \Sigma = \mathrm { D i a g } ( ( \sigma _ { i } ^ { 2 } ) _ { i \in [ d ] } ) , \mathrm { ~ w h e r e ~ } ( \sigma _ { i } ^ { 2 } ) _ { i \in [ d ] } \sim \mathrm { i n v } \chi ^ { 2 } ( 1 , 1 ) ^ { \otimes d } } \end{array} \right.\tag{9}
$$

<sub>w</sub>ith $\pmb { \mu }$ <sub>an</sub>d � i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>.</sub> W<sub>e</sub> <sub>now</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>our</sub> l<sub>ower</sub> b<sub>oun</sub>d f<sub>or</sub> <sub>pr</sub>i<sub>va</sub>t<sub>e</sub> <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>w</sub>ith <sub>un</sub>k<sub>nown</sub> dia<sub>g</sub>onal covariance.

Theorem 3.2. There exists $c > 0$ such that for all $d , n \in \mathbb { N }$ with � log $n \leq c { \sqrt { d } } ,$ , every $( 1 , c / n ) – D P$ algorithm A satisfies

$$
\operatorname* { P r } _ { X \sim \mathcal { N } ( \mu , \Sigma ) ^ { \otimes n } } \left[ \| \mathcal { A } ( X ) - \mu \| _ { 2 } \ge \frac { c d ^ { 3 / 2 } } { n ^ { 2 } \log d } \right] \ge \frac { 1 } { 4 } .
$$

That is, for all $\varepsilon \leq 1$ and $\textstyle { \delta \leq { \frac { c } { n } } }$ we have $\begin{array} { r } { \mathfrak { R } _ { \varepsilon , \delta } \big ( \mathcal { P } _ { d i a g } , n \big ) = \tilde { \Omega } \Big ( \frac { d ^ { 3 / 2 } } { n ^ { 2 } } \Big ) } \end{array}$

R<sub>eca</sub>ll th<sub>a</sub>t <sub>on</sub> i<sub>ns</sub>t<sub>ances</sub> $( \mu , \Sigma ) \sim { \mathcal { P } } _ { \mathrm { d i a g } }$ the known-covariance al<sub>g</sub>orithm of [DJYZZ24] obtains error

$$
\tilde { O } _ { \varepsilon , \delta } \Bigg ( \frac { d } { \sqrt { n } } \Bigg ) ,
$$

<sub>s</sub>in<sub>ce</sub> $\operatorname { t r } ( \Sigma ) = \Theta ( d ^ { 2 } )$ <sub>an</sub>d $\operatorname { t r } ( \Sigma ^ { 1 / 2 } ) = \Theta ( d \log d )$ <sub>w</sub>ith hi<sub>g</sub>h <sub>pro</sub>b<sub>a</sub>bilit<sub>y.</sub> Th<sub>us</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>sca</sub>l<sub>e-</sub>f<sub>ree</sub> “<sub>cos</sub>t” <sub>o</sub>f <sub>no</sub>t k<sub>now</sub>i<sub>ng</sub> th<sub>e covar</sub>i<sub>ance</sub> i<sub>n pr</sub>i<sub>va</sub>t<sub>e mean es</sub>ti<sub>ma</sub>ti<sub>on:</sub>

$$
\mathfrak { R } _ { \varepsilon , \delta } \big ( \mathcal { P } _ { \mathrm { d i a g } } , n \big ) = \tilde { \Omega } \left( 1 + \frac { \sqrt { d } } { n ^ { 3 / 2 } } \right) \cdot \mathfrak { R } _ { \varepsilon , \delta } ^ { \mathrm { k n o w n } } \big ( \mathcal { P } _ { \mathrm { d i a g } } , n \big ) .\tag{10}
$$

We note that the blow-u<sub>p</sub> in error ex<sub>p</sub>ressed b<sub>y</sub> (10), matches the error of the unknown covariance al<sub>g</sub>orithm from [DJYZZ24]. To be <sub>p</sub>recise, for the famil<sub>y</sub> $\mathcal { P } _ { \mathrm { d i a g } }$ an<sup>d</sup> <sub>p</sub>arameters $\varepsilon = 1 , \delta = c / n$ <sub>,</sub> th<sub>e</sub> method in [DJYZZ24, Theorem 4.1] achieves, with constant <sub>p</sub>robabilit<sub>y</sub>, error

$$
O \left( { \frac { d } { \sqrt { n } } } + { \frac { d \log d ( \log n ) ^ { 3 / 2 } } { n } } + { \frac { d ^ { 3 / 2 } ( \log d ) ^ { 2 } ( \log n ) ^ { 2 } \log ( d n ) } { n ^ { 2 } } } \right)
$$

f<sub>or</sub> $n \geq C { \sqrt { \log n } }$ l<sub>og</sub>� $\log ( d n )$ f<sub>or un</sub>i<sub>versa</sub>l <sub>cons</sub>t<sub>an</sub>t $C > 0$

Wh<sub>en</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e-s</sub>i<sub>ze</sub> i<sub>s</sub> l<sub>arge</sub> <sub>enoug</sub>h <sub>so</sub> th<sub>a</sub>t <sub>�</sub> l<sub>og</sub> $n \geq { \sqrt { d } } .$ <sub>,</sub> th<sub>e m</sub>iddl<sub>e</sub> t<sub>erm</sub> d<sub>om</sub>i<sub>na</sub>t<sub>es</sub> th<sub>e cos</sub>t <sub>o</sub>f <sub>p</sub>rivac<sub>y</sub>, and it is ti<sub>g</sub>ht u<sub>p</sub> to <sub>p</sub>ol<sub>y</sub>lo<sub>g</sub>arithmic factors, b<sub>y</sub> the known-covariance lower bound [DJYZZ24]. The remainin<sub>g</sub> sam<sub>p</sub>le-size re<sub>g</sub>ime of <sub>�</sub> lo<sub>g</sub> $n \leq { \sqrt { d } }$ <sub>,</sub> i<sub>s exac</sub>tl<sub>y</sub> th<sub>e reg</sub>i<sub>me w</sub>h<sub>ere</sub> th<sub>e</sub> thi<sub>r</sub>d t<sub>erm</sub> d<sub>om</sub>i<sub>na</sub>t<sub>es an</sub>d <sub>ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e reg</sub>i<sub>me o</sub>f <sub>our new</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>, s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t thi<sub>s</sub> t<sub>erm</sub> i<sub>s a</sub>l<sub>so</sub> i<sub>n</sub>d<sub>ee</sub>d necessar<sub>y,</sub> u<sub>p</sub> to <sub>p</sub>ol<sub>y</sub>lo<sub>g</sub>arithmic factors.

The <sub>p</sub>roof of Theorem 3.2 is similar to Theorem 2.1<sub>,</sub> but instead of reducin<sub>g</sub> from a sam<sub>p</sub>lin<sub>g</sub> <sub>pro</sub>bl<sub>em,</sub> <sub>we</sub> <sub>re</sub>d<sub>uce</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em:</sub>

Definition 3.3 $( ( k , \ell )$ -selection mechanism). Let $d , n , k , \ell \in \mathbb { N } .$ . An al<sub>g</sub>orithm M is a $( k , \ell )$ <sub>-se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> f<sub>or</sub> $\mathcal { P } _ { \mathrm { d i a g } }$ with sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> <sub>�</sub> if the out<sub>p</sub>ut $\pmb { \mathscr { s } } = \pmb { M } ( \pmb { X } ) \in \{ 0 , 1 \} ^ { d }$ <sub>o</sub>f th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es:</sub>

1. Boundedness: $\begin{array} { r } { \sum _ { j = 1 } ^ { d } s _ { j } \le \ell . } \end{array}$

2. Accuracy: Let π be the permutation such that $\pmb { \sigma } _ { \pi ( 1 ) } ^ { 2 } \geq . . . \geq \pmb { \sigma } _ { \pi ( d ) } ^ { 2 }$ <sub>.</sub> Th<sub>en</sub>

$$
\operatorname* { P r } _ { \Sigma \sim \mathcal { P } _ { \mathrm { d i a g } } } \left[ \sum _ { j = 1 } ^ { k } s _ { \pi ( j ) } \geq \frac { k } { 4 } \right] \geq \frac { 1 } { 3 } .
$$

I<sub>n</sub> <sub>o</sub>th<sub>er</sub> <sub>wor</sub>d<sub>s,</sub> <sub>a</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>s</sub> <sub>no</sub> <sub>more</sub> th<sub>an</sub> ℓ i<sub>n</sub>di<sub>ces,</sub> <sub>an</sub>d <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $1 / 3$ over th<sub>e</sub> <sub>ran</sub>d<sub>omness</sub> <sub>o</sub>f th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> <sub>an</sub>d th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m,</sub> it <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $k / 4$ i<sub>n</sub>di<sub>ces</sub> f<sub>rom</sub> th<sub>e</sub> i<sub>n</sub>di<sub>ces</sub> <sub>w</sub>ith th<sub>e</sub> � l<sub>arges</sub>t <sub>var</sub>i<sub>ances.</sub>

Th<sub>e</sub> <sub>nex</sub>t th<sub>eorem</sub> <sub>s</sub>h<sub>ows</sub> <sub>a</sub> t<sub>ra</sub>d<sub>eo</sub>f b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> <sub>accuracy</sub> <sub>an</sub>d th<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>ness</sub> <sub>o</sub>f <sub>pr</sub>i<sub>va</sub>t<sub>e</sub> $( k , \ell ) \cdot$ <sub>se</sub>l<sub>ec</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sms.</sub>

Theorem 3.4 (Lower bound for private selection). There exist constants $C , c > 0$ such that for all $d , k , n \in \mathbb { N }$ with � max $\left\{ n ^ { 2 } \log n , \log d \right\} \leq k \leq c d _ { \mathrm { \Delta } }$ , every $( k , \ell ) \mathop { - s e l e c t i o n }$ mechanism for $\mathcal { P } _ { d i a g }$ with sample complexity � that is $( 1 , c / n ) – D P$ must have $\ell \geq d / C \log ( d )$ .

Informall<sub>y</sub>, Theorem 3.4 states that a <sub>p</sub>rivate (�, ℓ)-selection mechanism which successfull<sub>y</sub> <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>a</sub> <sub>cons</sub>t<sub>an</sub>t f<sub>rac</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> $n ^ { 2 }$ l<sub>og �</sub> l<sub>arges</sub>t <sub>var</sub>i<sub>ance</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es,</sub> <sub>mus</sub>t <sub>a</sub>t th<sub>e</sub> <sub>same</sub> ti<sub>me</sub> <sub>se</sub>l<sub>ec</sub>t <sub>a</sub>l<sub>mos</sub>t <sub>a</sub>ll th<sub>e</sub> � <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>overa</sub>ll<sub>.</sub> Th<sub>e</sub> <sub>proo</sub>f i<sub>s</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> S<sub>ec</sub>ti<sub>on</sub> 3<sub>.</sub>2<sub>.</sub> N<sub>ex</sub>t<sub>,</sub> <sub>we</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> Th<sub>eorem</sub> 3<sub>.</sub>5<sub>,</sub> the reduction from <sub>p</sub>rivate selection, and com<sub>p</sub>lete the <sub>p</sub>roof of our main result (Theorem 3.2).

Lemma 3.5 (From mean estimation to selection). There exist constants $C , c > 0$ such that for all $\alpha > 0 _ { : }$ and $n , d , k \in \mathbb { N }$ with $C n \leq k \leq d ,$ , the following holds: $I f \mathcal { A }$ is $a \left( 1 , c / n \right) \cdot D P$ algorithm satisfying

$$
\mathop { \operatorname* { P r } } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { d i a g } } \left[ \| \mathcal { A } ( X ) - \mu \| _ { 2 } \leq \alpha \right] \geq \frac { 3 } { 4 } ,
$$

then, for $\textstyle \ell = \lceil \frac { C \alpha ^ { 2 } k n ^ { 2 } } { d ^ { 2 } } \rceil$ , there exists a $( 1 , c / n ) – D P$ mechanism M that is a $( k , \ell )$ -selection mechanism $f o r$ $\mathcal { P } _ { d i a g }$ with sample complexity �.

W<sub>e now com</sub>bi<sub>ne</sub> Th<sub>eorem</sub> 3<sub>.</sub>5 <sub>an</sub>d Th<sub>eorem</sub> 3<sub>.</sub>4 t<sub>o prove</sub> th<sub>e mean es</sub>ti<sub>ma</sub>ti<sub>on</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>.</sub>

ProofofTheorem 3.2. Let $c , C > 0$ <sub>an</sub>d $\varepsilon , \delta \in ( 0 , 1 )$ b<sub>e as</sub> i<sub>n</sub> Th<sub>eorem</sub> 3<sub>.</sub>4<sub>.</sub> Fi<sub>x</sub> $n \in \mathbb { N }$ <sub>an</sub>d <sub>se</sub>t $k = C \operatorname* { m a x } \left\{ n ^ { 2 } \log n , \log d \right\}$ <sup>. Su</sup>pp<sup>ose</sup> $\mathcal { A }$ i<sub>s a</sub>n $( \varepsilon , \delta ) \mathrm { - D P }$ al<sub>g</sub>orithm with sam<sub>p</sub>le com<sub>p</sub>lexit<sub>y</sub> $n ,$ th<sub>a</sub>t <sub>ac</sub>hi<sub>eves error � w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $2 / 3$ on instances sam<sub>p</sub>led from <sub>p</sub>rior $\mathcal { P } _ { \mathrm { d i a g . } }$ <sub>.</sub> Th<sub>en,</sub> b<sub>y</sub> Th<sub>eorem</sub> 3<sub>.</sub>5<sub>,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s an</sub> $\left( \varepsilon , \delta \right) \mathrm { - D P ~ } ( k , \ell )$ <sub>-se</sub>l<sub>ec</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm w</sub>ith $\ell \leq C \alpha ^ { 2 } k n ^ { 2 } / d ^ { 2 }$ . But b<sub>y</sub> our settin<sub>g</sub> of �<sub>,</sub> Theorem 3.4 im<sub>p</sub>lies that $\ell \geq d / C \log d ,$ <sub>, an</sub>d th<sub>ere</sub>f<sub>ore</sub> $\alpha ^ { 2 } \geq c ^ { 2 } d ^ { 3 } / C k n ^ { 2 }$ lo<sub>g</sub>� ≥ $\Omega ( d ^ { 3 } / n ^ { 4 } \log ( d ) ^ { 2 } )$ □

## 3.1 Reduction from selection

In this section<sub>,</sub> we com<sub>p</sub>lete the <sub>p</sub>roof of Theorem 3.5 b<sub>y</sub> usin<sub>g</sub> a <sub>p</sub>rivate mean estimation al<sub>g</sub>orithm to desi<sub>g</sub>n a $( k , \ell )$ <sub>-se</sub>l<sub>ec</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm.</sub> I<sub>n</sub>t<sub>u</sub>iti<sub>ve</sub>l<sub>y,</sub> th<sub>e se</sub>l<sub>ec</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm se</sub>l<sub>ec</sub>t<sub>s</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>es</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub> <sub>su</sub>f<sub>ers</sub> <sub>error</sub> <sub>a</sub>b<sub>ove</sub> <sub>a</sub> <sub>care</sub>f<sub>u</sub>ll<sub>y</sub> <sub>c</sub>h<sub>osen</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>,</sub> <sub>an</sub>d <sub>among</sub> th<sub>ese,</sub> <sub>we</sub> <sub>expec</sub>t t<sub>o</sub> fi<sub>n</sub>d <sub>a</sub> l<sub>o</sub>t <sub>o</sub>f th<sub>e coor</sub>di<sub>na</sub>t<sub>es w</sub>ith l<sub>arge var</sub>i<sub>ance.</sub>

Proof of Theorem 3.5. Let $\mathcal { A }$ b<sub>e</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>w</sub>ith <sub>accuracy</sub> <sub>�</sub> <sub>as</sub> i<sub>n</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t <sub>o</sub>f Th<sub>eorem</sub> 3<sub>.</sub>5<sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> th<sub>e se</sub>l<sub>ec</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm</sub> $\mathcal { M }$ <sub>w</sub>ith fi<sub>xe</sub>d <sub>parame</sub>t<sub>ers</sub> $\begin{array} { r } { \tau = \frac { d } { C n \sqrt { k } } } \end{array}$ <sub>an</sub>d $\rho = d$ <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

$$
\begin{array} { r l } & { \mathcal M ( X ) : \mathrm { s a m p l e } \mu \sim \mathcal N ( 0 , \rho ^ { 2 } \mathbb I _ { d } ) \mathrm { ~ a n d ~ s e t ~ } Y = X + \mu . } \\ & { \qquad \mathrm { F o r ~ } j \in [ d ] \mathrm { ~ s e t ~ } s _ { j } = \mathbb { 1 } \big [ \big | \mathcal { H } ( Y ) _ { j } - \mu _ { j } \big | \ge \tau \big ] . } \\ & { \qquad \mathrm { I f ~ } \sum s _ { j } \le \ell \mathrm { ~ r e t u r n ~ } s = ( s _ { 1 } , \ldots , s _ { d } ) , \mathrm { a n d ~ o t h e r w i s e ~ r e t u r n ~ } \mathbf 0 . } \end{array}\tag{11}
$$

To show that M is a (�, ℓ)-selection mechanism, we must show that it never selects more than ℓ <sub>coor</sub>di<sub>na</sub>t<sub>es,</sub> <sub>an</sub>d th<sub>a</sub>t <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 / 4$ it <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $k / 4$ <sub>o</sub>f th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>w</sub>ith th<sub>e</sub> � lar<sub>g</sub>est variances. B<sub>y</sub> ins<sub>p</sub>ection of (11), one can easil<sub>y</sub> see that M never out<sub>p</sub>uts a selection of <sub>more</sub> th<sub>an</sub> ℓ <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub>

Let $J = \{ \pi ( 1 ) , \ldots , \pi ( k ) \}$ b<sub>e</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>es o</sub>f th<sub>e</sub> l<sub>arges</sub>t � <sub>var</sub>i<sub>ances</sub> i<sub>n</sub> �<sub>.</sub> I<sub>n</sub> th<sub>e rema</sub>i<sub>n</sub>d<sub>er</sub> of the <sub>p</sub>roof, we ar<sub>g</sub>ue that M out<sub>p</sub>uts at least $k / 4$ of the coordinates in J with <sub>p</sub>robabilit<sub>y</sub> at least $1 / 4 .$ First we ar<sub>g</sub>ue that M t<sub>yp</sub>icall<sub>y</sub> does not return 0: B<sub>y</sub> construction of $M _ { i }$ <sub>,</sub> f<sub>or</sub> <sub>any</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f $\tau ,$ <sub>we</sub> h<sub>ave</sub> $\begin{array} { r } { \tau ^ { 2 } \sum _ { j = 1 } ^ { d } s _ { j } \le \| \mathcal { A } ( \dot { Y } ) - \mu \| _ { 2 } ^ { 2 } } \end{array}$ <sub>,</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore,</sub> <sub>se</sub>tti<sub>ng</sub> $\ell = \lceil \alpha ^ { 2 } / \tau ^ { 2 } \rceil$ and usin<sub>g</sub> the accurac<sub>y</sub> <sub>g</sub>uarantee of A,

$$
\operatorname* { P r } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { \mathrm { d i a g } } } \left[ \sum _ { j = 1 } ^ { d } s _ { j } \le \ell \right] \ge \operatorname* { P r } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { \mathrm { d i a g } } } \left[ \left. \mathcal { A } ( Y ) - \mu \right. \le \alpha \right] \ge \frac { 3 } { 4 } .\tag{12}
$$

Next<sub>,</sub> we ar<sub>g</sub>ue that $s _ { j } = 1$ f<sub>or</sub> <sub>a</sub>t l<sub>eas</sub>t $k / 4$ of the indices in J with <sub>p</sub>robabilit<sub>y</sub> at least $1 / 4$ <sub>.</sub> O<sub>u</sub>r <sub>proo</sub>f l<sub>everages</sub> <sub>a</sub> <sub>vers</sub>i<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d l<sub>ower</sub> b<sub>oun</sub>d f<sub>or</sub> <sub>pr</sub>i<sub>va</sub>t<sub>e</sub> <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> th<sub>a</sub>t b<sub>oun</sub>d<sub>s</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>coor</sub>di<sub>na</sub>t<sub>es w</sub>ith <sub>error a</sub>b<sub>ove some</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub>

Proposition 3.6 (Coordinate-wise error lower bound [GLSUZ26]). There exist constants $C , c > 0$ such thatfor all $n , d \in \mathbb { N }$ and $\alpha , \rho > 0$ satisfying

$$
\rho \geq C \operatorname * { m a x } \{ 1 , \alpha \} , \qquad d \geq C ( n + \alpha ^ { 2 } n ^ { 2 } ) ,
$$

$i f \mu \sim { \cal N } ( 0 , \rho ^ { 2 } \mathbb { I } _ { d } )$ and $X = x _ { 1 } , . . . , x _ { n } \sim N ( \pmb { \mu } , \mathbb { I } _ { d } ) ^ { \otimes n }$ , then every $( 1 , c / n ) – D P$ mechanism A satisfies

$$
\operatorname* { P r } _ { \mu \sim N ( 0 , \rho ^ { 2 } \mathbb { I } _ { d } ) } \left[ \left| \left\{ j \in [ d ] : | \mathcal { A } ( X ) _ { j } - \mu _ { j } | \ge \alpha \right\} \right| \ge d / 2 \right] \ge \frac { 2 } { 3 } .
$$

Si<sub>nce</sub> Th<sub>eorem</sub> 3<sub>.</sub>6 i<sub>s</sub> f<sub>or</sub> id<sub>en</sub>tit<sub>y covar</sub>i<sub>ance, a</sub> f<sub>ew ex</sub>t<sub>ra s</sub>t<sub>eps w</sub>ill b<sub>e nee</sub>d<sub>e</sub>d b<sub>e</sub>f<sub>ore we can</sub> a<sub>pp</sub>l<sub>y</sub> it to our settin<sub>g</sub>. Informall<sub>y,</sub> we will construct an al<sub>g</sub>orithm $\mathcal { A } _ { \Sigma }$ th<sub>a</sub>t <sub>uses</sub> $\mathcal { A }$ <sub>as a su</sub>b<sub>rou</sub>ti<sub>ne</sub> f<sub>or</sub> <sub>a</sub> �<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em</sub> <sub>w</sub>ith id<sub>en</sub>tit<sub>y</sub> <sub>covar</sub>i<sub>ance.</sub> W<sub>e</sub> th<sub>en</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>per-coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>error</sub> <sub>o</sub>f $\mathcal { A } _ { \Sigma }$ <sub>can</sub> b<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>error</sub> <sub>o</sub>f $\mathcal { A }$ on the coordinates in J. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Th<sub>eorem</sub> 3<sub>.</sub>6 t<sub>o</sub> $\mathcal { A } _ { \Sigma }$ allows us to conclude that A must have lar e error in man of the coordinates in J.

C<sub>ons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> �<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em:</sub>

$$
\theta \sim { \cal N } \bigg ( 0 , \frac { 4 k ^ { 2 } \rho ^ { 2 } } { d ^ { 2 } } \mathbb { I } _ { k } \bigg ) \mathrm { a n d } Z = z _ { 1 } , . . . , z _ { n } \sim { \cal N } ( \theta , \mathbb { I } _ { k } ) ^ { \otimes n } .\tag{13}
$$

For a vector $y \in \mathbb { R } ^ { d }$ <sub>an</sub>d $J \subseteq [ d ]$ <sub>,</sub> l<sub>e</sub>t $y _ { J }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e vec</sub>t<sub>or</sub> $( y _ { j } ) _ { j \in J }$ <sub>, an</sub>d <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>l<sub>y,</sub> f<sub>or a ma</sub>t<sub>r</sub>i<sub>x</sub> $\Sigma \in \mathbb { R } ^ { d \times d }$ <sub>,</sub> l<sub>e</sub>t $\Sigma _ { J }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> $| J | \times | J |$ m<sub>a</sub>trix <sub>w</sub>ith <sub>e</sub>ntri<sub>es</sub> $\Sigma _ { j , j ^ { \prime } }$ f<sub>or</sub> $j , j ^ { \prime } \in J$ <sub>.</sub> D<sub>e</sub>fi<sub>ne a</sub>l<sub>gor</sub>ith<sub>m</sub> $\mathcal { A } _ { \Sigma }$ as f<sub>o</sub>ll<sub>ows:</sub>

$$
\begin{array} { l } { { \displaystyle \mathcal { R } _ { \Sigma } ( Z ) : { \mathrm { I f } \sigma } _ { \pi ( k ) } ^ { 2 } < \frac { d ^ { 2 } } { 4 k ^ { 2 } } \mathrm { ~ r e t u r n ~ \bot . ~ O t h e r w i s e ~ c o n t i n u e . } } } \\ { { \mathrm { S a m p l e ~ } \theta _ { \overline { { J } } } ^ { \prime } \sim \mathcal { N } ( 0 , \rho ^ { 2 } \mathbb { I } _ { d - k } ) \mathrm { ~ a n d ~ f o r ~ e a c h ~ } i \in [ n ] : } } \\ { { \mathrm { S a m p l e ~ } \tilde { y } _ { i , J } = \frac { d } { 2 k } z _ { i } + w _ { i } \mathrm { ~ w h e r e ~ } w _ { i } \sim \mathcal { N } \bigg ( 0 , \Sigma _ { J } - \frac { d ^ { 2 } } { 4 k ^ { 2 } } \mathbb { I } _ { k } \bigg ) . } } \\ { { \mathrm { S a m p l e ~ } \tilde { y } _ { i , \overline { { J } } } \sim \mathcal { N } \bigg ( \theta _ { \overline { { J } } } ^ { \prime } , \Sigma _ { \overline { { J } } } \bigg ) } } \\ { { \mathrm { R e t u r n ~ } \frac { 2 k } { d } \mathcal { R } ( \tilde { Y } ) _ { J } } } \end{array}\tag{14}
$$

T<sub>o see</sub> th<sub>a</sub>t $\mathcal { A } _ { \Sigma }$ is <sub>p</sub>ri<sub>v</sub>ate<sub>,</sub> obser<sub>v</sub>e that since $\mathcal { A }$ i<sub>s</sub> $( \varepsilon , \delta ) \mathrm { - D P }$ <sub>, an</sub>d f<sub>or</sub> fi<sub>xe</sub>d <sub>ran</sub>d<sub>om co</sub>i<sub>ns,</sub> $\mathcal { A } _ { \Sigma }$ s<sup>i</sup>m<sub>p</sub><sup>l</sup><sub>y</sub> runs $\mathcal { A }$ <sub>on</sub> <sub>an</sub> <sub>a</sub>fi<sub>ne</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> <sub>an</sub>d <sub>ou</sub>t<sub>pu</sub>t<sub>s</sub> th<sub>e</sub> <sub>resu</sub>lt<sub>,</sub> <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t $\mathcal { A } _ { \Sigma }$ i<sub>s</sub> <sub>a</sub>l<sub>so</sub> $( \varepsilon , \delta ) \mathrm { \cdot D P }$

N<sub>ex</sub>t<sub>, we re</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e accuracy o</sub>f $\mathcal { A } _ { \Sigma }$ on the task defined in (13), to the accurac<sub>y</sub> of A on the coordinates in J in the ex<sub>p</sub>eriment in the statement of the lemma. Let � denote the event that $\sigma _ { \pi ( k ) } ^ { 2 } \geq d ^ { 2 } / 4 k ^ { 2 }$ <sub>, an</sub>d l<sub>e</sub>t $\begin{array} { r } { \dot { \theta _ { J } ^ { \prime } } = \frac { d } { 2 k } \theta . } \end{array}$ <sub>.</sub> Th<sub>en, con</sub>diti<sub>ona</sub>l <sub>on</sub> � <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t $\tilde { \boldsymbol { y } } _ { i , J } \sim \mathcal { N } \big ( \theta _ { J } ^ { \prime } , \Sigma _ { J } \big )$ <sub>an</sub>d th<sub>us,</sub> b<sub>y</sub> th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f $\tilde { y } _ { i , \overline { { J } } }$ <sub>,</sub> th<sub>a</sub>t $\tilde { Y } \sim N \Big ( ( \theta _ { J } ^ { \prime } , \theta _ { \overline { { J } } } ^ { \prime } ) , \Sigma \Big )$ . Additionall<sub>y</sub>, b<sub>y</sub> the distribution of θ <sub>an</sub>d $\theta _ { \overline { { J } } } ^ { \prime }$ <sub>, we</sub> h<sub>ave</sub> th<sub>a</sub>t $\theta ^ { \prime } = ( \theta _ { J } ^ { \prime } , \theta _ { \overline { { J } } } ^ { \prime } ) \sim N ( 0 , \rho ^ { 2 } \mathbb { \tilde { L } } _ { d } )$ <sub>.</sub> F<sub>ur</sub>th<sub>ermore, con</sub>diti<sub>ona</sub>l <sub>on</sub> $G _ { i }$ <sub>,</sub> f<sub>or eac</sub>h $j \in [ k ]$ <sub>we</sub> h<sub>ave</sub>

$$
\left| \mathcal { A } _ { \Sigma } ( Z ) _ { j } - \theta _ { j } \right| = \frac { 2 k } { d } \Big | \mathcal { R } ( \tilde { Y } ) _ { \pi ( j ) } - \theta _ { \pi ( j ) } ^ { \prime } \Big | \equiv \frac { 2 k } { d } \big | \mathcal { A } ( Y ) _ { \pi ( j ) } - \mu _ { \pi ( j ) } \big | ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e equa</sub>lit<sub>y</sub> i<sub>n</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> h<sub>o</sub>ld<sub>s s</sub>i<sub>nce con</sub>diti<sub>ona</sub>l <sub>on</sub> � <sub>we</sub> h<sub>ave</sub> $\boldsymbol { Y } \equiv \tilde { \boldsymbol { Y } }$ <sub>an</sub>d $\mu \equiv \theta ^ { \prime }$ <sub>.</sub> Th<sub>us,</sub> if $B _ { 0 }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e even</sub>t th<sub>a</sub>t $\left| \left\{ j \in [ k ] : | \mathcal { A } _ { \Sigma } ( Z ) _ { j } - \pmb { \theta } _ { j } | \leq \tau \cdot \frac { 2 k } { d } \right\} \right| \geq k / 2$ <sub>an</sub>d $B _ { 1 }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e even</sub>t th<sub>a</sub>t $\left| \left\{ j \in [ k ] : | \mathcal { A } ( \pmb { Y } ) _ { \pi ( j ) } - \dot { \mu } _ { \pi ( j ) } | \leq \tau \right\} \right| \geq k / 2$ th<sub>en</sub>

$$
\mathrm { P r } [ B _ { 1 } ] \leq \mathrm { P r } [ { \overline { { G } } } ] + \mathrm { P r } [ B _ { 0 } ] .
$$

B<sub>y</sub> our settin<sub>g</sub> of $\dot { \mathbf { \rho } } _ { \rho }$ <sub>an</sub>d <sub>�</sub> <sub>we</sub> h<sub>ave</sub> $2 k \tau / d = 2 \sqrt { k } / C n \leq 2 k \rho / C d$ (the standard deviation in (13) is $2 k \rho / d )$ <sub>, an</sub>d b<sub>y our se</sub>tti<sub>ng o</sub>f� <sub>an</sub>d <sub>�, we</sub> h<sub>ave</sub> $\textstyle k \geq C { \Bigl ( } n + { \frac { 4 k ^ { 2 } \tau ^ { 2 } n ^ { 2 } } { d ^ { 2 } } } { \Bigr ) }$ . Th<sub>u</sub>s<sub>, w</sub>e can a<sub>pp</sub>l<sub>y</sub> Theorem 3.6 to $\mathcal { A } _ { \Sigma }$ <sub>w</sub>ith <sub>am</sub>bi<sub>en</sub>t di<sub>mens</sub>i<sub>on</sub> � <sub>an</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>e-w</sub>i<sub>se error</sub> th<sub>res</sub>h<sub>o</sub>ld $\textstyle { \boldsymbol { \alpha } } \gets { \boldsymbol { \tau } } \cdot { \frac { 2 k } { d } }$ t<sub>o see</sub> th<sub>a</sub>t $\begin{array} { r } { \mathrm { P r } [ B _ { 0 } ] \leq \frac { 1 } { 3 } } \end{array}$ <sub>.</sub> Additi<sub>ona</sub>ll<sub>y, s</sub>i<sub>nce</sub> b<sub>y</sub> Th<sub>eorem</sub> A<sub>.</sub>4 <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t $\textstyle \operatorname* { P r } [ { \overline { { G } } } ] \leq { \frac { 2 } { k } }$ <sub>, we can conc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>a</sub>t P<sub>r</sub> $\begin{array} { r } { [ B _ { 1 } ] \leq \frac { 1 } { 3 } + \frac { 1 } { 1 2 } } \end{array}$ f<sub>or</sub> $k \geq 2 4 -$ <sub>—</sub>th<sub>a</sub>t i<sub>s,</sub> ${ \mathcal { A } } ( Y )$ h<sub>as error a</sub>t <sub>mos</sub>t <sub>�</sub> i<sub>n a</sub>t l<sub>eas</sub>t $\frac { k } { 2 }$ of the coordinates in J with <sub>p</sub>robabilit<sub>y</sub> at most $\textstyle { \frac { 1 } { 3 } } + { \frac { 1 } { 1 2 } }$

We now return to the anal<sub>y</sub>sis of mechanism M (defined in (11)). A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the union bound over $B _ { 1 }$ and the event controlled in (12), we see that with <sub>p</sub>robabilit<sub>y</sub> at least $\textstyle { \frac { 1 } { 4 } }$ over the sam<sub>p</sub>lin<sub>g</sub> <sub>o</sub>f th<sub>e</sub> i<sub>ns</sub>t<sub>ance</sub> $( \mu , \Sigma ) \sim { \mathcal P } _ { \mathrm { d i a g } }$ <sub>,</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> $ { \boldsymbol { X } } \sim  { \mathcal { N } } ( 0 ,  { \boldsymbol { \Sigma } } )$ , and the coins of the mechanism, M out<sub>p</sub>uts a vector s such that $\sum s _ { i } \leq \ell ,$ <sub>an</sub>d $s _ { j } = 1$ f<sub>or a</sub>t l<sub>eas</sub>t $k / 2$ <sub>o</sub>f th<sub>e coor</sub>di<sub>na</sub>t<sub>es</sub> $j \in J$ . Thus, M is a $( k , \ell )$ <sub>se</sub>l<sub>ec</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm w</sub>ith $\ell = \lceil \alpha ^ { 2 } / \tau ^ { 2 } \rceil$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h<sub>,</sub> b<sub>y</sub> <sub>our</sub> <sub>se</sub>tti<sub>ng</sub> <sub>o</sub>f $\tau ,$ <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub> □

## 3.2 Private selection lower bound

To <sub>p</sub>ro<sub>v</sub>e Theorem 3.4<sub>,</sub> similarl<sub>y</sub> to the dia<sub>g</sub>onal case<sub>,</sub> <sub>w</sub>e consider a test statistic that meas<sub>u</sub>res the correlation between the selection mechanism M and the em<sub>p</sub>irical covariance of the dataset. Fix <sub>a cova</sub>ri<sub>a</sub>n<sub>ce</sub> m<sub>a</sub>trix $\Sigma \in \mathbb { R } ^ { d \times d } = \operatorname { D i a g } ( \sigma _ { 1 } ^ { 2 } , . . . , \sigma _ { d } ^ { 2 } )$ <sub>, a se</sub>l<sub>ec</sub>ti<sub>on vec</sub>t<sub>or</sub> $s \in \{ 0 , 1 \} ^ { d }$ <sub>,</sub> and a test <sub>p</sub>oint $z \in \mathbb { R } ^ { d }$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> t<sub>es</sub>t

$$
\mathcal { T } _ { \Sigma , s } ( z ) : = \sum _ { j \in [ d ] } \mathcal { T } _ { \Sigma , s , j } ( z ) , \quad \mathrm { w h e r e } \quad \mathcal { T } _ { \Sigma , s , j } ( z ) : = \left( \frac { z _ { j } ^ { 2 } } { \sigma _ { j } ^ { 2 } } - 1 \right) \operatorname* { m i n } \left\{ \sigma _ { j } , L \right\} s _ { j } ,\tag{15}
$$

<sub>an</sub>d <sub>w</sub>h<sub>ere</sub> $L = L ( d , k )$ i<sub>s a parame</sub>t<sub>er w</sub>hi<sub>c</sub>h <sub>we w</sub>ill <sub>se</sub>t l<sub>a</sub>t<sub>er.</sub>

We t<sub>yp</sub>icall<sub>y</sub> omit Σ and � from the notation for readabilit<sub>y</sub>, usin<sub>g</sub> just $\mathcal { T }$ <sub>an</sub>d ${ \mathcal { T } } _ { j } .$ Additi<sub>ona</sub>ll<sub>y,</sub> f<sub>or</sub> <sub>conven</sub>i<sub>ence,</sub> <sub>we</sub> <sub>w</sub>ill <sub>use</sub> th<sub>e</sub> <sub>s</sub>h<sub>or</sub>th<sub>an</sub>d $h ( \sigma )$ f<sub>or</sub> <sub>m</sub>i<sub>n</sub> $\{ \sigma , L \}$ . Similarl<sub>y</sub> to Section 2<sub>, w</sub>e com<sub>p</sub>are the di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n <sub>o</sub>f th<sub>e</sub> t<sub>wo s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs w</sub>ith <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> $\Sigma \sim { \mathcal { P } } _ { \mathrm { d i a g } }$ <sub>,</sub> d<sub>a</sub>t<sub>ase</sub>t $\boldsymbol { X } \sim \boldsymbol { N } ( 0 , \boldsymbol { \Sigma } ) ^ { \otimes n }$ <sub>,</sub> f<sub>res</sub>h <sub>samp</sub>l<sub>e</sub> $\boldsymbol { z } \sim \mathcal { N } ( 0 , \Sigma )$ <sub>, an</sub>d <sub>se</sub>l<sub>ec</sub>ti<sub>on vec</sub>t<sub>or</sub> $\pmb { s } \sim { \mathcal { M } } ( { \pmb X } )$

• The OUT sample case in which we consider $\mathcal { T } _ { \Sigma , s } ( z )$

• The IN sample case in which we consider $\mathcal { T } _ { \Sigma , s } ( \pmb { x } _ { i } )$ with i uniform in [�].

Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a</sub> th<sub>res</sub>h<sub>o</sub>ld $\tau \in \mathsf { \Gamma } ( 0 , 1 )$ <sub>suc</sub>h th<sub>a</sub>t <sub>w</sub>ith hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub> $\mathcal T ( { \pmb x } _ { i } ) \geq d \tau / n$ <sub>an</sub>d $\mathcal { T } ( z ) < d \tau / n$ <sub>.</sub> W<sub>e</sub> d<sub>e</sub>f<sub>er</sub> th<sub>e proo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 3<sub>.</sub>7 t<sub>o</sub> S<sub>ec</sub>ti<sub>on</sub> 3<sub>.</sub>2<sub>.</sub>1<sub>, an</sub>d Theorem 3.8 to Section 3.2.2, and com<sub>p</sub>lete the <sub>p</sub>roof of the selection lower bound (Theorem 3.4) b<sub>e</sub>l<sub>ow.</sub>

Lemma 3.7 (OUT sample test is small). Fix $\tau \in ( 0 , 1 )$ . There exist $C > 0$ such thatfor all �, �, $n , \ell , \in \mathbb { N }$ with $\begin{array} { r } { k \geq C n ^ { 2 } \log n , i f L = \frac { d } { 4 k } } \end{array}$ then every $( k , \ell )$ -selection mechanism M satisfies

$$
\operatorname* { P r } _ { \Sigma , X , z , \mathcal { M } } \left[ \mathcal { T } ( z ) \geq \frac { d \tau } { n } \right] = O \left( \frac { 1 } { n ^ { 2 } } \right) .
$$

Lemma 3.8 (IN sample test is large). Let $C > 0$ be suficiently large. There exists $\tau \in ( 0 , 1 )$ such that for all $d , k , n , \ell \in \mathbb { N }$ with � max $\{ n , \log d \} \leq k \leq d / C$ and $\begin{array} { r } { \ell \leq d / C \log d , i f L = \frac { d } { 4 k } } \end{array}$ then every $( k , \ell )$ -selection mechanism M satisfies

$$
\operatorname* { P r } _ { \Sigma , X , i , \mathcal { M } } \left[ \mathcal { T } ( \boldsymbol { x } _ { i } ) \geq \frac { d \tau } { n } \right] = \Omega \left( \frac { 1 } { n } \right) .
$$

ProofofTheorem 3.4. The prooffollows by contradiction. Suppose M is an $\left( \varepsilon , \delta \right) \mathrm { - D P } \left( k , \ell \right)$ <sub>-se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm w</sub>ith $\varepsilon \leq 1 , \delta \leq c / n , k \geq C$ max $\left\{ n ^ { 2 } \log n , \log d \right\}$ <sub>, an</sub>d $\ell \leq d / C \log ( d )$

F<sub>o</sub>r <sub>eve</sub>r<sub>y</sub> ind<sub>e</sub>x $i \in [ n ]$ l<sub>e</sub>t $\pmb { X } ^ { \oplus i }$ denote X with $x _ { i }$ re<sub>p</sub>laced b<sub>y</sub> z, and let ${ \mathcal { T } } ^ { \oplus i }$ d<sub>eno</sub>t<sub>e</sub> $\mathcal { T }$ <sub>execu</sub>t<sub>e</sub>d <sub>on</sub> $M ( X ^ { \oplus i } )$ . Then, since M is $( \varepsilon , \delta ) \mathrm { - D P }$ <sub>we</sub> h<sub>ave</sub>

$$
\mathcal { T } ( z ) \approx _ { \varepsilon , \delta } \mathcal { T } ^ { \oplus i } ( z ) \equiv \mathcal { T } ( x _ { i } )
$$

<sub>w</sub>h<sub>ere</sub> $\approx _ { \varepsilon , \delta }$ i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> Th<sub>eorem</sub> 2<sub>.</sub>10<sub>, an</sub>d th<sub>e equa</sub>lit<sub>y</sub> i<sub>n</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ and z are both sam<sub>p</sub>led from $N ( 0 , \Sigma )$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y o</sub>f th<sub>e o</sub>th<sub>er</sub> d<sub>a</sub>t<sub>apo</sub>i<sub>n</sub>t<sub>s.</sub> H<sub>ence,</sub> if $i \sim [ n ]$ <sub>an</sub>d <sub>�</sub> i<sub>s</sub> th<sub>e cons</sub>t<sub>an</sub>t i<sub>n</sub> Th<sub>eorem</sub> 3<sub>.</sub>8<sub>,</sub> th<sub>en</sub>

$$
\Omega \left( \frac { 1 } { n } \right) \leq \operatorname* { P r } \left[ \mathcal { T } ( x _ { i } ) \geq \frac { d \tau } { n } \right] \leq e ^ { \varepsilon } \operatorname* { P r } \left[ \mathcal { T } ( z ) \geq \frac { d \tau } { n } \right] + \delta \leq O \left( \frac { 1 } { n ^ { 2 } } \right) + \frac { c } { n } .
$$

<sub>w</sub>hi<sub>c</sub>h<sub>,</sub> f<sub>or su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y sma</sub>ll <sub>� y</sub>i<sub>e</sub>ld<sub>s a con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on.</sub>

## 3.2.1 Analysis of the test in the OUT case

In this section<sub>,</sub> <sub>w</sub>e <sub>p</sub>ro<sub>v</sub>e Theorem 3.7<sub>,</sub> <sub>w</sub>hich states that $\mathcal { T } ( z ) \geq d \tau / n$ with <sub>p</sub>robabilit<sub>y</sub> at most $O ( 1 / n ^ { 2 } )$ .

Proof of Theorem 3.7. First we bound the expectation. Observe that for every fixed $\Sigma , X ,$ <sub>,</sub> <sub>an</sub>d $s = \mathcal { M } ( \boldsymbol { X } )$

$$
\underset { z _ { j } } { \mathbb { E } } [ \mathcal { T } _ { j } ( z ) ] = \underset { z _ { j } } { \mathbb { E } } \Bigg [ \frac { z _ { j } ^ { 2 } } { \sigma _ { j } ^ { 2 } } - 1 \Bigg ] h ( \sigma _ { j } ) s _ { j } = 0
$$

and thus<sub>,</sub> b<sub>y</sub> linearit<sub>y</sub> of ex<sub>p</sub>ectation $\mathbb { E } [ \mathcal { T } ( z ) ] = 0 .$

Next<sub>,</sub> we <sub>p</sub>rove a hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub> bound on $\mathcal T ( z )$ <sub>.</sub> Si<sub>nce</sub> f<sub>or</sub> fi<sub>xe</sub>d $\sigma _ { j }$ <sub>an</sub>d $z _ { j } \sim \mathcal { N } ( 0 , \sigma _ { j } ^ { 2 } )$ <sub>,</sub> th<sub>e</sub> <sub>ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>e</sub> $z _ { j } ^ { 2 } / \sigma _ { j } ^ { 2 }$ i<sub>s</sub> $\chi _ { 1 } ^ { 2 } .$ , we can a<sub>pp</sub>l<sub>y</sub> a scalar Bernstein ine<sub>q</sub>ualit<sub>y</sub> (e.<sub>g</sub>., Corollar<sub>y</sub> 2.9.2 [Ver26]) to see that

$$
\operatorname* { P r } \left[ \left| \sum _ { j \in [ d ] } \mathcal { T } _ { j } ( z ) \right| \geq \sqrt { \sum _ { j \in [ d ] } h ( \sigma _ { j } ) ^ { 2 } \log ( 1 / \beta ) s _ { j } } + L \log ( 1 / \beta ) \right] \leq \beta .
$$

Next<sub>,</sub> we <sub>p</sub>rove a hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub> bound on $\textstyle \sum h ( \sigma _ { j } ) ^ { 2 } s _ { j }$

The <sub>p</sub>roof <sub>g</sub>oes throu<sub>g</sub>h a Chernof bound for bounded nonne<sub>g</sub>ative random variables (Theorem $\mathrm { A } . 2 )$ a<sub>pp</sub>lied to $h ( \sigma _ { j } ) ^ { 2 } \in [ 0 , L ^ { 2 } ]$ <sub>.</sub> W<sub>e</sub> fi<sub>rs</sub>t b<sub>oun</sub>d th<sub>e expec</sub>t<sub>a</sub>ti<sub>on:</sub>

$$
\mathbb { E } [ h ( \sigma _ { j } ) ^ { 2 } ] \le \int _ { 0 } ^ { s ^ { 2 } } \operatorname* { P r } [ \sigma _ { j } ^ { 2 } \ge t ] d t \le 1 + \int _ { 1 } ^ { L ^ { 2 } } t ^ { - 1 / 2 } d t \le 2 L
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Theorem A.2<sub>,</sub> there exists a constant $C _ { 1 }$ <sub>suc</sub>h th<sub>a</sub>t

$$
\sqrt { \sum _ { j \in [ d ] } h ( \sigma _ { j } ) ^ { 2 } \log ( 1 / \beta ) } \leq \sqrt { C _ { 1 } L d \log ( 1 / \beta ) } + L \log ( 1 / \beta )
$$

<sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \beta _ { : }$ <sub>, w</sub>hi<sub>c</sub>h<sub>, s</sub>i<sub>nce</sub> $s _ { j } \in \{ 0 , 1 \}$ i<sub>mp</sub>li<sub>es</sub> th<sub>e same</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e sum</sub> i<sub>nvo</sub>l<sub>v</sub>i<sub>ng</sub> $s _ { j } .$ . Settin<sub>g</sub> $\beta = O ( 1 / n ^ { 2 } )$ <sub>an</sub>d <sub>c</sub>h<sub>oos</sub>i<sub>ng</sub> $C = C ( \tau , C _ { 1 } )$ <sub>, we see</sub> th<sub>a</sub>t if $L = d / 4 k \le d / C n ^ { 2 }$ <sup>l</sup>o<sub>g</sub> � we <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d b<sub>oun</sub>d

$$
\operatorname* { P r } \left[ \sum _ { j \in [ d ] } \mathcal { T } _ { j } ( z ) \geq \frac { d \tau } { n } \right] \leq O \left( \frac { 1 } { n ^ { 2 } } \right) .
$$

## 3.2.2 Analysis of the test in the IN case

Thi<sub>s sec</sub>ti<sub>on</sub> i<sub>s</sub> d<sub>evo</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e proo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 3<sub>.</sub>8<sub>, w</sub>hi<sub>c</sub>h <sub>s</sub>t<sub>a</sub>t<sub>es</sub> th<sub>a</sub>t $\mathcal T ( { \pmb x } _ { i } ) \geq d \tau / n$ with <sub>p</sub>robabilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $\Omega ( 1 / n )$ <sub>.</sub> Th<sub>roug</sub>h<sub>ou</sub>t thi<sub>s</sub> <sub>sec</sub>ti<sub>on</sub> <sub>we</sub> l<sub>e</sub>t $\mathbb { E } [ \mathcal { T } _ { j } ( \pmb { x } _ { i } ) \ | \ \pmb { X } , s ]$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>ran</sub>d<sub>om</sub> <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> <sub>g</sub>i<sub>ven</sub> b<sub>y</sub> sam<sub>p</sub>lin<sub>g</sub> a realization � and � of X and $s ,$ and returnin<sub>g</sub> $\operatorname { \mathbb { E } } [ \mathcal { T } _ { j } ( x _ { i } ) \mid X = X , s = s ]$ <sub>.</sub> Additi<sub>ona</sub>ll<sub>y,</sub> for random variables w and $k ,$ l<sub>e</sub>t $\scriptstyle w \mid _ { k }$ denote the random variable <sub>g</sub>iven b<sub>y</sub> sam<sub>p</sub>lin<sub>g</sub> a realization � of k and then sam<sub>p</sub>lin<sub>g</sub> a realization � from the distribution of w conditioned on the event that $k = k$ <sub>.</sub> B<sub>e</sub>f<sub>ore</sub> f<sub>orma</sub>ll<sub>y</sub> <sub>s</sub>t<sub>a</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>ms</sub> <sub>an</sub>d <sub>proo</sub>f<sub>s,</sub> <sub>we</sub> <sub>prov</sub>id<sub>e</sub> <sub>some</sub> i<sub>n</sub>t<sub>u</sub>iti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> <sub>o</sub>f $\mathbb { E } [ \mathcal { T } _ { j } ( { \pmb x } _ { i } ) \ | \ { \pmb X } , s ]$ f<sub>or</sub> fi<sub>xe</sub>d $j \in [ d ]$ . It wi<sup>ll</sup> <sup>b</sup>e convenient to use t<sup>h</sup>e <sup>f</sup>o<sup>ll</sup>owing conjugate prior <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> of the scaled-inverse chi-s<sub>q</sub>uared distribution:

Fact 3.9 (Conjugate prior property). Let $\pmb { \sigma } ^ { 2 } \sim \mathrm { i n v } \chi ^ { 2 } ( \nu , \tau ^ { 2 } )$ , and $X = ( x _ { 1 } , . . . , x _ { n } ) \sim N ( 0 , \pmb { \sigma } ^ { 2 } ) ^ { \otimes n }$ then it holds that:

$$
\pmb { \sigma } ^ { 2 } | _ { X } \sim \mathsf { i n v } \chi ^ { 2 } \left( \nu + n , \frac { \nu \tau ^ { 2 } + \sum _ { i \in [ n ] } \pmb { x } _ { i } ^ { 2 } } { \nu + n } \right) .
$$

Let $\begin{array} { r } { \pmb q _ { j } = \frac { 1 } { n } \sum _ { i \in [ n ] } \pmb x _ { i , j } ^ { 2 } } \end{array}$ denote the em<sub>p</sub>irical variance of X in coordinate $j .$ Sin<sub>ce</sub> $\pmb { \sigma } _ { j } ^ { 2 } \sim \mathrm { i n v } \chi ^ { 2 } ( 1 , 1 )$ b<sub>y</sub> Th<sub>eorem</sub> $3 . 9 ,$ conditional on X we have $\begin{array} { r } { \pmb { \sigma } _ { j } ^ { 2 } \sim \operatorname { i n v } \chi ^ { 2 } \left( n + 1 , \frac { 1 + n \pmb q _ { j } } { n + 1 } \right) } \end{array}$ . Assumin<sub>g �</sub> is lar<sub>g</sub>e<sub>,</sub> we have th<sub>a</sub>t $\begin{array} { r } { \sigma _ { j } ^ { 2 } \approx \frac { 1 + n q _ { j } } { 1 + n } } \end{array}$ with hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub>. Thus<sub>,</sub> we ex<sub>p</sub>ect that

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { T } _ { j } ( x _ { i } ) \mid X , s ] = \mathbb { E } \left[ \left( \frac { \mathbb { E } _ { i } [ x _ { i , j } ^ { 2 } ] } { \sigma _ { j } ^ { 2 } } - 1 \right) h ( \sigma _ { j } ) s _ { j } \Bigg | X , s \right] } \\ & { \quad \quad \approx \displaystyle \frac { q _ { j } - 1 } { 1 + n q _ { j } } \operatorname* { m i n } \left( \sqrt { \frac { 1 + n q _ { j } } { 1 + n } } , L \right) s _ { j } , } \end{array}
$$

where the first e<sub>q</sub>ualit<sub>y</sub> follows since i is inde<sub>p</sub>endent of X and �, and the second e<sub>q</sub>ualit<sub>y</sub> follows <sub>s</sub>in<sub>ce</sub> $\mathbb { E } [ { \pmb x } _ { i , j } ^ { 2 } \ \mid \ { \pmb X } ] = { \pmb q } _ { j }$ <sub>,</sub> and b<sub>y</sub> substitutin<sub>g</sub> the a<sub>pp</sub>roximation $\frac { 1 + n q _ { j } } { 1 + n }$ f<sub>or</sub> $\sigma _ { j } ^ { 2 }$ . Ins<sub>p</sub>ectin<sub>g</sub> the ex<sub>p</sub>ression on the ri<sub>g</sub>ht hand side, we see that (i<sub>g</sub>norin<sub>g</sub> constant factors) there are two salient r<sub>eg</sub>im<sub>es</sub>: Fir<sub>s</sub>t<sub>,</sub> <sub>w</sub>h<sub>e</sub>n $q _ { j } \gg 1$ <sub>,</sub> the ex<sub>p</sub>ectation is at least min ${ \left( \sqrt { q _ { j } } , L \right) } / n$ <sub>, an</sub>d <sub>secon</sub>d<sub>, w</sub>h<sub>en</sub> $q _ { j } \ll 1$ al<sub>g</sub>ebraic mani<sub>p</sub>ulation shows that the ex<sub>p</sub>ectation is alwa<sub>y</sub>s at least $- { \sqrt { q _ { j } } } / n$ <sub>.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>expec</sub>t th<sub>e expec</sub>t<sub>a</sub>ti<sub>on</sub> t<sub>o</sub> h<sub>ave</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or:</sub>

$$
\operatorname { \mathbb { E } } [ { \mathcal { T } } _ { j } ( x _ { i } ) \mid X , s ] \approx { \frac { s _ { j } } { n } } \cdot { \left\{ \begin{array} { l l } { \operatorname* { m i n } \left( { \sqrt { q _ { j } } } , L \right) } & { q _ { j } > 1 } \\ { - { \sqrt { q _ { j } } } } & { q _ { j } \leq 1 } \end{array} \right. }
$$

Su<sup>b</sup>ject to t<sup>h</sup>e a<sup>b</sup>ove <sup>h</sup>euristic, we see t<sup>h</sup>at <sup>f</sup>or eac<sup>h</sup> coor<sup>d</sup>inate � wit<sup>h</sup> ${ q } _ { j } \leq 1$ <sub>,</sub> the ex<sub>p</sub>ectation is alwa<sub>y</sub>s <sub>a</sub>t l<sub>eas</sub>t $- 1 / n$ <sub>.</sub> Sin<sub>ce</sub> $M$ <sub>can se</sub>l<sub>ec</sub>t <sub>a</sub>t <sub>mos</sub>t ℓ <sub>coor</sub>di<sub>na</sub>t<sub>es,</sub> th<sub>ese coor</sub>di<sub>na</sub>t<sub>es can con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e a</sub> t<sub>o</sub>t<sub>a</sub>l <sub>o</sub>f $\frac { - \ell } { n }$ t<sub>o</sub> th<sub>e sum.</sub> O<sub>n</sub> th<sub>e o</sub>th<sub>er</sub> h<sub>an</sub>d<sub>,</sub> if th<sub>e coor</sub>di<sub>na</sub>t<sub>es w</sub>ith th<sub>e</sub> � l<sub>arges</sub>t $\mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta } } \mathbf { \delta } _ { \mathbf { \eta \delta } }$ <sub>are a</sub>ll <sub>a</sub>t l<sub>eas</sub>t �<sub>,</sub> th<sub>en,</sub> b<sub>ecause</sub> $M$ i<sub>s requ</sub>i<sub>re</sub>d t<sub>o se</sub>l<sub>ec</sub>t <sub>a</sub>t l<sub>eas</sub>t <sub>a cons</sub>t<sub>an</sub>t f<sub>rac</sub>ti<sub>on o</sub>f th<sub>em,</sub> th<sub>ey mus</sub>t <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e a</sub>t l<sub>eas</sub>t $\frac { k L } { n }$ t<sub>o</sub> th<sub>e sum.</sub> It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t th<sub>e expec</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> l<sub>arge w</sub>h<sub>enever</sub> $k L \gg \ell \approx d / \log d ,$ <sub>,</sub> or e<sub>q</sub>uivalentl<sub>y</sub> <sub>w</sub>h<sub>en</sub> $L \approx d / k$ . Usin<sub>g</sub> the distribution of $\sigma _ { j }$ <sub>an</sub>d $\boldsymbol { X }$ <sub>, one can s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> � l<sub>arges</sub>t $\mathbf { \mu } _ { q _ { j } }$ <sub>are</sub> i<sub>n</sub>d<sub>ee</sub>d t<sub>yp</sub>icall<sub>y</sub> at least $d / k _ { : }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> <sub>proo</sub>f<sub>.</sub>

Th<sub>e</sub> fi<sub>rs</sub>t <sub>s</sub>t<sub>ep</sub> i<sub>n</sub> <sub>our</sub> <sub>proo</sub>f i<sub>s</sub> Th<sub>eorem</sub> 3<sub>.</sub>10<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>prov</sub>id<sub>es</sub> <sub>a</sub> <sub>more</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> f<sub>orm</sub> f<sub>or</sub> <sub>ana</sub>l<sub>yz</sub>i<sub>ng</sub> th<sub>e</sub> ex<sub>p</sub>ectation of the test conditional on X and s. While the heuristic anal<sub>y</sub>sis above <sub>p</sub>rovides intuition b<sub>y</sub> reasonin<sub>g</sub> about the em<sub>p</sub>irical variance $\mathbf { \mu } _ { q j }$ <sub>,</sub> Theorem 3.10 <sub>p</sub>rovides an e<sub>q</sub>uivalent re<sub>p</sub>resentation of the ex<sub>p</sub>ectation over $\sigma _ { j }$ th<sub>a</sub>t <sub>ex</sub>hibit<sub>s</sub> th<sub>e same qua</sub>lit<sub>a</sub>ti<sub>ve</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or as</sub> th<sub>e</sub> h<sub>eur</sub>i<sub>s</sub>ti<sub>c.</sub>

Claim 3.10 (Simplifying the expectation). For all $j \in [ d ]$ , the conditional expectation o $f \mathcal { T } _ { j }$ satisfies

$$
\operatorname { \mathbb { E } } [ { \mathcal { T } } _ { j } ( x _ { i } ) \mid X , s ] = { \frac { 1 } { n } } \cdot \operatorname { \mathbb { E } } [ ( L \mathbb { 1 } [ \sigma _ { j } > L ] - { \frac { h ( \sigma _ { j } ) } { \sigma _ { j } ^ { 2 } } } ) s _ { j } ] X , s ] .
$$

In particular,

$$
\operatorname { \mathbb { E } } [ { \mathcal { T } } ( x _ { i } ) \mid X , s ] = { \frac { 1 } { n } } \cdot \operatorname { \mathbb { E } } [ \sum _ { j \in [ d ] } ( L \mathbb { 1 } [ \sigma _ { j } > L ] - { \frac { h ( \sigma _ { j } ) } { \sigma _ { j } ^ { 2 } } } ) s _ { j }  X , s ] .
$$

Proof. Since � is independent of s conditional on $X ,$ <sub>,</sub> <sub>we</sub> fi<sub>rs</sub>t <sub>ana</sub>l<sub>yze</sub> th<sub>e</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> <sub>over</sub> � conditional on X, and then include the conditionin<sub>g</sub> on s at the end of the <sub>p</sub>roof. Let $q _ { j } ~ =$ $\textstyle { \frac { 1 } { n } } \sum _ { i \in [ n ] } x _ { i , j } ^ { 2 }$ <sup>b</sup>e t<sup>h</sup>e empirica<sup>l</sup> variance in coor<sup>d</sup>inate �. T<sup>h</sup>e proo<sup>f</sup> <sup>f</sup>o<sup>ll</sup>ows <sup>f</sup>rom integration <sup>b</sup>y <sub>par</sub>t<sub>s.</sub> I<sub>n</sub> <sub>w</sub>h<sub>a</sub>t f<sub>o</sub>ll<sub>ows,</sub> <sub>we</sub> <sub>cons</sub>id<sub>er</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> $j \in [ d ]$ , an<sup>d</sup> omit t<sup>h</sup>e su<sup>b</sup>script � <sup>f</sup>rom t<sup>h</sup>e <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons</sub> t<sub>o</sub> <sub>re</sub>d<sub>uce</sub> <sub>no</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>c</sub>l<sub>u</sub>tt<sub>er.</sub> W<sub>e</sub> b<sub>eg</sub>i<sub>n</sub> b<sub>y</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>equa</sub>ti<sub>on:</sub>

$$
\mathbb { E } [ \sigma h ^ { \prime } ( \sigma ) \mid X ] = \mathbb { E } [ h ( \sigma ) \middle ( ( n + 1 ) - \frac { 1 + n q } { \sigma ^ { 2 } } ) \Bigg | X ] .\tag{16}
$$

R<sub>eca</sub>ll b<sub>y</sub> Th<sub>eorem</sub> 3<sub>.</sub>9<sub>,</sub> th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f $\sigma ^ { 2 } | _ { X }$ h<sub>as</sub> d<sub>ens</sub>it<sub>y</sub> $\begin{array} { r } { p _ { \pmb { \sigma } ^ { 2 } } ( t ) \propto t ^ { - ( \nu / 2 + 1 ) } \exp \left( \frac { - \nu \tau ^ { 2 } } { 2 t } \right) } \end{array}$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\nu = n + 1$ <sub>an</sub>d $\textstyle \tau ^ { 2 } = { \frac { 1 + n q } { 1 + n } }$ . Since the <sub>q</sub>uantit<sub>y</sub> in the ex<sub>p</sub>ectation in <sub>q</sub>uestion is σ (as o<sub>pp</sub>osed to $\sigma ^ { 2 } )$ <sub>,</sub> it <sub>w</sub>ill b<sub>e conven</sub>i<sub>en</sub>t t<sub>o wor</sub>k <sub>w</sub>ith th<sub>e</sub> d<sub>ens</sub>it<sub>y o</sub>f $\sigma | _ { X }$ in<sub>s</sub>t<sub>ea</sub>d<sub>.</sub> L<sub>e</sub>t $f _ { \sigma } ( s )$ b<sub>e</sub> th<sub>e</sub> d<sub>ens</sub>it<sub>y o</sub>f $\sigma | _ { X }$ <sub>.</sub> Th<sub>en,</sub> b<sub>y a c</sub>h<sub>ange o</sub>f <sub>measure, we</sub> h<sub>ave</sub> th<sub>a</sub>t $\begin{array} { r } { f _ { \sigma } ( s ) = 2 s p _ { \sigma ^ { 2 } } ( s ^ { 2 } ) \propto s ^ { - ( \nu + 1 ) } \exp \left( \frac { - \nu \tau ^ { 2 } } { 2 s ^ { 2 } } \right) } \end{array}$ . Now<sub>,</sub> <sup>we</sup> <sup>a</sup>pp<sup>l</sup>y <sup>inte</sup>g<sup>ration</sup> <sup>b</sup>y p<sup>arts</sup> <sup>to</sup> <sup>the</sup> <sup>ex</sup>p<sup>ectation:</sup>

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \sigma h ^ { \prime } ( \sigma ) \mid X ] = \int _ { s = 0 } ^ { \infty } s h ^ { \prime } ( s ) f _ { \sigma } ( s ) d s } \\ { = [ s f _ { \sigma } ( s ) h ( s ) ] _ { 0 } ^ { \infty } - \int _ { s = 0 } ^ { \infty } h ( s ) ( s f _ { \sigma } ( s ) ) ^ { \prime } d s . } \end{array}
$$

Sin<sub>ce</sub> $h ( s ) \leq L$ <sub>an</sub>d $\begin{array} { r } { \operatorname* { l i m } _ { s \to \infty } s f _ { \sigma } ( s ) = \operatorname* { l i m } _ { s \to 0 } s f _ { \sigma } ( s ) = 0 , } \end{array}$ <sub>,</sub> th<sub>e</sub> b<sub>oun</sub>d<sub>ary</sub> t<sub>erm</sub> i<sub>s</sub> 0<sub>.</sub> T<sub>o</sub> h<sub>an</sub>dl<sub>e</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> t<sub>erm,</sub> <sub>we</sub> <sub>use</sub> th<sub>e</sub> f<sub>ormu</sub>l<sub>a</sub> f<sub>or</sub> $f _ { \sigma } ( s )$ to ex<sub>p</sub>licitl<sub>y</sub> com<sub>p</sub>ute the derivative:

$$
( s f _ { \sigma } ( s ) ) ^ { \prime } = f _ { \sigma } ( s ) \left( \frac { 1 + n q } { s ^ { 2 } } - ( n + 1 ) \right) .
$$

Substitutin<sub>g</sub> in the inte<sub>g</sub>ral $\begin{array} { r } { \int h ( s ) ( s f _ { \sigma } ( s ) ) ^ { \prime } } \end{array}$ �� <sub>y</sub>ields (16). B<sub>y</sub> al<sub>g</sub>ebra, we see that

$$
\big ( n + 1 \big ) - \frac { 1 + n q } { \sigma ^ { 2 } } = \Bigg ( 1 - \frac { 1 } { \sigma ^ { 2 } } \Bigg ) - n \bigg ( \frac { q } { \sigma ^ { 2 } } - 1 \bigg ) ,
$$

<sub>an</sub>d th<sub>us,</sub> th<sub>a</sub>t

$$
n \mathbb { E } \bigg [ \bigg ( \frac { q } { \sigma ^ { 2 } } - 1 \bigg ) h ( \sigma ) \biggm | X \bigg ] = \mathbb { E } \bigg [ h ( \sigma ) - \sigma h ^ { \prime } ( \sigma ) - \frac { h ( \sigma ) } { \sigma ^ { 2 } } \biggm | X \bigg ] .
$$

Th<sub>e s</sub>t<sub>a</sub>t<sub>emen</sub>t f<sub>o</sub>ll<sub>ows s</sub>i<sub>nce</sub> $h ( \sigma ) - \sigma h ^ { \prime } ( \sigma ) = L 1 1 [ \sigma > L ]$ <sub>,</sub> <sub>an</sub>d <sub>s</sub>i<sub>nce</sub>

$$
\mathbb { E } \left[ \left( \frac { q _ { j } } { \sigma _ { j } ^ { 2 } } - 1 \right) h ( \sigma _ { j } ) \Bigg | X \right] s _ { j } = \mathbb { E } \left[ \operatorname* { l i m } _ { i \sim [ n ] } \left[ \left( \frac { \boldsymbol { x } _ { i , j } ^ { 2 } } { \sigma _ { j } ^ { 2 } } - 1 \right) h ( \sigma _ { j } ) \right] \Bigg | X \right] s _ { j } = \mathbb { E } [ \mathcal { T } _ { j } ( x _ { i } ) \mid X , s ] ,
$$

<sub>w</sub>hi<sub>c</sub>h h<sub>o</sub>ld<sub>s</sub> b<sub>ecause</sub> $i \sim [ n ]$ <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> <sub>an</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>o</sub>f $X , s ,$ <sub>an</sub>d $\Sigma .$

I<sub>n or</sub>d<sub>er</sub> t<sub>o s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e con</sub>diti<sub>ona</sub>l <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> l<sub>arge, we mus</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e pos</sub>iti<sub>ve</sub> <sub>p</sub>art of the ex<sub>p</sub>ression in Theorem 3.10 is much lar<sub>g</sub>er than the ne<sub>g</sub>ative <sub>p</sub>art. In line with our h<sub>eur</sub>i<sub>s</sub>ti<sub>c</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>on,</sub> Th<sub>eorem</sub> 3<sub>.</sub>11 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>w</sub>h<sub>en</sub> $L = d / 4 k$ <sub>,</sub> the <sub>p</sub>ositi<sub>v</sub>e <sub>p</sub>art is $\Omega ( d )$ <sub>w</sub>ith <sub>cons</sub>t<sub>an</sub>t <sub>p</sub>robabilit<sub>y</sub>.

Claim 3.11 (Lower bounding the positive part). There exists $c > 0$ such that $i f L = d / 4 k$ and � ∈ [max(6 lo<sub>g</sub> �, 24), ��] then

$$
\mathbb { E } \left[ \left. \sum _ { j \in [ d ] } L \mathbb { 1 } [ \pmb { \sigma } _ { j } > L ] \pmb { s } _ { j } \right| \pmb { X } , \pmb { s } \right] \geq c d
$$

with probability at least �.

Before divin<sub>g</sub> in to the <sub>p</sub>roof<sub>,</sub> we <sub>g</sub>ive a brief overview of the main ideas. Let $\pmb { g } = \sum L \mathbb { 1 } [ \pmb { \sigma } _ { j } > L ] \pmb { s } _ { j }$ <sub>an</sub>d $\mu _ { X , s } = \operatorname { \mathbb { E } } [ \pmb { \mathscr { g } } \mid X , s ]$ <sub>.</sub> Th<sub>en</sub> Th<sub>eorem</sub> 3<sub>.</sub>11 <sub>s</sub>t<sub>a</sub>t<sub>es</sub> th<sub>a</sub>t $\mu _ { X , s } \geq$ <sub>�</sub>� <sub>w</sub>ith <sub>cons</sub>t<sub>an</sub>t <sub>pro</sub>b<sub>a</sub>bilit<sub>y.</sub> Th<sub>e</sub> <sub>essence o</sub>f th<sub>e proo</sub>f i<sub>s</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>ree o</sub>b<sub>serva</sub>ti<sub>ons:</sub> Fi<sub>rs</sub>t<sub>,</sub> $\pmb { g }$ i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>a</sub>b<sub>ove</sub> b<sub>y</sub> ��<sub>.</sub> S<sub>econ</sub>d<sub>,</sub> the ex<sub>p</sub>ectation $\mathbb { E } [ \mu _ { X , s } ] = \mathbb { E } [ \pmb { g } ]$ i<sub>s</sub> in th<sub>e</sub> int<sub>e</sub>r<sub>va</sub>l $[ c d , 2 d ]$ <sub>.</sub> A<sub>n</sub>d thi<sub>r</sub>d<sub>, s</sub>i<sub>nce</sub> $\pmb { g }$ i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>a</sub>b<sub>ove</sub> b<sub>y a</sub> <sub>sum o</sub>f i<sub>.</sub>i<sub>.</sub>d<sub>.</sub> B<sub>ernou</sub>lli <sub>ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>es,</sub> $\operatorname* { P r } [ g \ge 4 d ] \le e ^ { - d / 3 L }$ <sub>.</sub> T<sub>a</sub>k<sub>en</sub> t<sub>oge</sub>th<sub>er,</sub> th<sub>ese o</sub>b<sub>serva</sub>ti<sub>ons</sub> im<sub>p</sub>l<sub>y</sub> that $\mu _ { X , s }$ <sub>canno</sub>t <sub>p</sub>l<sub>ace</sub> t<sub>oo</sub> <sub>muc</sub>h <sub>mass</sub> <sub>a</sub>b<sub>ove</sub> 4�<sub>,</sub> <sub>an</sub>d<sub>,</sub> <sub>s</sub>i<sub>nce</sub> $\mathbb { E } [ \mu _ { X , s } ] \in [ c d , 2 d ]$ <sub>,</sub> it must <sub>p</sub>lace <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>mass a</sub>b<sub>ove</sub> $c d / 2 .$

Proof. As in the proof sketch above, let $\pmb { g } = \sum L \mathbb { 1 } [ \pmb { \sigma } _ { j } > L ] \pmb { s } _ { j }$ <sub>an</sub>d $\mu _ { X , s } = \operatorname { \mathbb { E } } [ g \mid X , s ]$ <sub>.</sub> W<sub>e</sub> fi<sub>rs</sub>t <sub>prove</sub> th<sub>e</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>upper</sub> b<sub>oun</sub>d f<sub>o</sub>ll<sub>ows</sub> <sub>s</sub>i<sub>nce</sub> $\pmb { \mathscr { s } } _ { j } = \mathscr { M } ( \pmb { X } ) _ { j } \leq 1$ <sub>,</sub> <sub>an</sub>d <sub>s</sub>i<sub>nce</sub> $\operatorname* { P r } [ \sigma _ { j } \geq L ] \leq 2 / L$ (because $\sigma ^ { 2 } \sim \mathsf { i n v } \chi ^ { 2 } ( 1 , 1 ) )$ <sub>.</sub> T<sub>o see w</sub>h<sub>y</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d h<sub>o</sub>ld<sub>s, o</sub>b<sub>serve</sub> th<sub>a</sub>t b<sub>y</sub> Th<sub>eorem</sub> ${ \mathrm { A . 4 } }$ <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t $\sigma _ { ( k ) } \geq d / \vert $ 4� <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - 2 / k$ . Let π be the <sub>p</sub>ermutation orderin<sub>g</sub> $( \sigma _ { j } ^ { 2 } ) _ { j \in [ d ] }$ <sub>.</sub> Sin<sub>ce</sub> $s _ { \pi ( j ) } = 1$ f<sub>or a</sub>t l<sub>eas</sub>t $k / 4$ <sub>o</sub>f th<sub>e</sub> $j \in [ k ]$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 / 3$ <sub>,</sub> th<sub>e</sub> <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\sum _ { j \in \left[ k \right] } L \mathbb { 1 } \left[ \sigma _ { \pi ( j ) } > L \right] s _ { \pi ( j ) } \geq \sum _ { j \in \left[ k \right] } \frac { d } { 4 k } s _ { \pi ( j ) } \geq \frac { d } { 1 6 }
$$

<sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 / 4$ f<sub>or</sub> $k \geq 2 4$ <sub>, an</sub>d th<sub>us,</sub> $\mathbb { E } [ \mu _ { X , s } ] = \mathbb { E } [ g ] \in [ c d , 2 d ]$ f<sub>or some cons</sub>t<sub>an</sub>t $c > 0$

Next, we <sub>p</sub>rove a high <sub>p</sub>robabilit<sub>y</sub> bound on g. Observe that $\begin{array} { r } { g \le \sum L \mathbb { 1 } [ \pmb { \sigma } _ { j } > L ] } \end{array}$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s a</sub> <sub>sum o</sub>f i<sub>.</sub>i<sub>.</sub>d<sub>.</sub> B<sub>ernou</sub>lli <sub>ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>es w</sub>ith <sub>mean a</sub>t <sub>mos</sub>t 2�<sub>.</sub> L<sub>e</sub>t $B _ { 1 }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t $g \geq 4 d _ { \mathrm { i } }$ <sub>an</sub>d l<sub>e</sub>t $B _ { 2 }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t $\mathbb { E } [ \mathbb { 1 } [ B _ { 1 } ] \ | \ X , s ] \ge e ^ { - d / 6 L }$ <sub>.</sub> B<sub>y a</sub> Ch<sub>erno</sub>f b<sub>oun</sub>d<sub>,</sub> $\operatorname* { P r } [ B _ { 1 } ] \leq e ^ { - d / 3 L }$ <sub>an</sub>d th<sub>us,</sub> b<sub>y</sub> M<sub>ar</sub>k<sub>ov</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y,</sub> $\mathbb { E } [ \mathbb { 1 } [ B _ { 2 } ] ] \le e ^ { - d / 6 L }$ . In other words, g is rarel<sub>y</sub> bigger than $4 d ,$ and with hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub> over X and s, random variable $g | _ { X , s }$ is rarel<sub>y</sub> bi<sub>gg</sub>er than 4�.

We now ar<sub>g</sub>ue that $\mu _ { X , s }$ <sub>mus</sub>t b<sub>e</sub> <sub>a</sub>t l<sub>eas</sub>t $\Omega ( d )$ with constant <sub>p</sub>robabilit<sub>y</sub> over X and s.

$$
\begin{array} { r l } & { c d \leq \mathbb { E } \left[ \mu _ { X , s } \right] \leq \mathbb { E } [ \mu _ { X , s } \mathbb { 1 } [ \overline { { B _ { 2 } } } ] ] + L d e ^ { - d / 6 L } } \\ & { \qquad \leq \mathbb { E } \left[ \mathbb { E } [ g \mathbb { 1 } [ \overline { { B } } _ { 1 } ] \mid X , s ] \right] + \underbrace { L d \mathbb { E } \big [ \mathbb { E } [ \mathbb { 1 } [ B _ { 1 } ] \mid X , s ] \mathbb { 1 } [ \overline { { B _ { 2 } } } ] \big ] } _ { \leq L d e ^ { - d / 6 L } } + L d e ^ { - d / 6 L } } \\ & { \qquad \leq \mathbb { E } [ \mu _ { X , s } \mathbb { 1 } [ \mu _ { X , s } < c ^ { \prime } d ] ] + 4 d \mathbb { E } [ \mathbb { 1 } [ \mu _ { X , s } > c ^ { \prime } d ] ] + 2 L d e ^ { - k / 6 } \mathrm { f o r ~ a n y ~ } c ^ { \prime } > 0 } \\ & { \qquad \leq 2 c ^ { \prime } d + 4 d \mathbb { P r } [ \mu _ { X , s } > c ^ { \prime } d ] , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>na</sub>l i<sub>nequa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s s</sub>i<sub>nce</sub> $L = d / 4 k$ <sub>an</sub>d $k \geq 6 \log ( d )$ . Settin<sub>g</sub> $c ^ { \prime } = c / 4$ an<sup>d</sup> rearran<sub>g</sub><sup>i</sup>n<sub>g</sub> t<sub>erms, we see</sub> th<sub>a</sub>t $\mu _ { X , s } \geq c d / 4$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $c / 8$ □

To com<sub>p</sub>lete o<sub>u</sub>r anal<sub>y</sub>sis of the conditional ex<sub>p</sub>ectation<sub>,</sub> <sub>w</sub>e <sub>p</sub>ro<sub>v</sub>e Theorem 3.12<sub>,</sub> <sub>w</sub>hich states th<sub>a</sub>t th<sub>e nega</sub>ti<sub>ve</sub> t<sub>erms o</sub>f th<sub>e con</sub>diti<sub>ona</sub>l <sub>expec</sub>t<sub>a</sub>ti<sub>on canno</sub>t d<sub>ecrease</sub> th<sub>e overa</sub>ll <sub>sum</sub> b<sub>y muc</sub>h <sub>more</sub> th<sub>an</sub> ${ \sqrt { \ell d } } .$ <sub>.</sub> W<sub>e no</sub>t<sub>e</sub> th<sub>a</sub>t thi<sub>s ana</sub>l<sub>ys</sub>i<sub>s</sub> di<sub>verges s</sub>li<sub>g</sub>htl<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> h<sub>eur</sub>i<sub>s</sub>ti<sub>c</sub> b<sub>oun</sub>d <sub>o</sub>f ℓ d<sub>er</sub>i<sub>ve</sub>d <sub>a</sub>t th<sub>e</sub> <sub>s</sub>t<sub>ar</sub>t <sub>o</sub>f th<sub>e sec</sub>ti<sub>on;</sub> h<sub>owever,</sub> th<sub>e</sub> d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>s m</sub>i<sub>nor—</sub>f<sub>or</sub> $\ell \approx d / \log d$ <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t ${ \sqrt { \ell d } } \approx \ell { \sqrt { \log d } } .$

Claim 3.12 (Upper bounding the negative part). For all $C > 0$

$$
\mathbb { E } \left[ \sum _ { j \in [ d ] } { \frac { h ( \sigma _ { j } ) } { \sigma _ { j } ^ { 2 } } } s _ { j } \right] X , s \Bigg ] \leq C { \sqrt { \ell d } }
$$

with probability at least $1 - 1 / C .$

Proof. Since $h ( \sigma _ { j } ) \leq \sigma _ { j }$ <sub>,</sub> <sub>an</sub>d $\mathbb { E } [ \sigma _ { i } ^ { - 2 } ] < \infty ,$ <sub>, we can</sub> b<sub>oun</sub>d th<sub>e expec</sub>t<sub>a</sub>ti<sub>on o</sub>f th<sub>e nega</sub>ti<sub>ve</sub> t<sub>erms</sub> over X and s usin<sub>g</sub> Cauch<sub>y</sub>-Schwarz and Jensen’s ine<sub>q</sub>ualit<sub>y</sub>

$$
\mathbb { E } \Bigg [ \sum \frac { h ( \sigma _ { j } ) } { \sigma _ { j } ^ { 2 } } s _ { j } \Bigg ] \leq \mathbb { E } \Bigg [ \sqrt { \ell \sum \sigma _ { j } ^ { - 2 } } \Bigg ] \leq \sqrt { \ell \sum \mathbb { E } [ \sigma _ { j } ^ { - 2 } ] } = \sqrt { \ell d } ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>na</sub>l i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>ecause</sub> $\sigma _ { j } ^ { - 2 }$ f<sub>o</sub>ll<sub>ows a</sub> $\chi _ { 1 } ^ { 2 }$ distribution. B<sub>y</sub> Markov’s ine<sub>q</sub>ualit<sub>y</sub> over X and s, the ne<sub>g</sub>ative terms are at most $C \sqrt { \ell d }$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - 1 / C$ □

Th<sub>eorems</sub> 3<sub>.</sub>10 t<sub>o</sub> 3<sub>.</sub>12 <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e con</sub>diti<sub>ona</sub>l <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> $\Omega ( d / n )$ with constant <sub>p</sub>robabilit<sub>y</sub>. However, in order to show that the test itself is lar<sub>g</sub>e with constant <sub>p</sub>robabilit<sub>y</sub> (over the randomness of both � and X), we re<sub>q</sub>uire a bound on the conditional variance. Theorem 3.13 states that the variance of the test conditional on X is t<sub>yp</sub>icall<sub>y</sub> less than $L d / n$

Claim 3.13 (Bounding the variance). For all $C > 0$

$$
\operatorname { V a r } [ \mathbb { E } [ { \mathcal { T } } ( x _ { i } ) \mid \Sigma ] \mid X , s ] \leq { \frac { 4 C L d } { n } }
$$

with probability at least $1 - 1 / C$

Proof. Since the ${ \boldsymbol { \sigma } } _ { i } ^ { 2 } { \boldsymbol { \mathfrak { s } } }$ <sub>s are samp</sub>l<sub>e</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> f<sub>or eac</sub>h $j \in [ d ]$ , we have that conditioned on X and s, the terms $\check { \mathcal { T } } _ { j } ( x _ { i } )$ <sub>are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>.</sub> Th<sub>us,</sub>

$$
\begin{array} { r l r } & { } & { \underset { \ b { \Sigma } } { \mathrm { V a r } } \Bigg [ \underset { i } { \mathbb { E } } \left[ \mathcal { T } ( x _ { i } ) \mid \Sigma \right] \Bigg | \textbf {  { X } } , s \Bigg ] = \underset { j \in [ d ] } { \sum } \mathrm { V a r } \Bigg [ \Bigg ( \frac { q _ { j } } { \sigma _ { j } ^ { 2 } } - 1 \Bigg ) h ( \sigma _ { j } ) s _ { j } \Bigg | \textbf {  { X } } , s \Bigg ] } \\ & { } & { \leq \underset { j \in [ d ] } { \sum } \underset { \Sigma } { \mathbb { E } } \Bigg [ \Bigg ( \frac { q _ { j } } { \sigma _ { j } ^ { 2 } } - 1 \Bigg ) ^ { 2 } h ( \sigma _ { j } ) ^ { 2 } \Bigg | \textbf {  { X } } \Bigg ] . } \end{array}
$$

Takin<sub>g</sub> the ex<sub>p</sub>ectation over X and s and swa<sub>pp</sub>in<sub>g</sub> the order of ex<sub>p</sub>ectation we see that,

$$
\begin{array} { r l } & { \displaystyle \sum _ { \boldsymbol { X } , s } \left[ \mathrm { V a r } \left[ \mathbb { E } [ \mathcal { T } ( \boldsymbol { x } _ { i } ) \mid \boldsymbol { \Sigma } ] \right] \left. \boldsymbol { X } , s \right] \right] \leq \displaystyle \sum _ { j \in [ d ] } \mathbb { E } \mathbb { E } \left[ \left( \frac { q _ { j } } { \sigma _ { j } ^ { 2 } } - 1 \right) ^ { 2 } h ( \sigma _ { j } ) ^ { 2 } \Bigg \vert \boldsymbol { \Sigma } \right] } \\ & { \leq \displaystyle \sum _ { j \in [ d ] } \mathbb { E } \left[ \frac { 2 } { n } \cdot h ( \sigma _ { j } ) ^ { 2 } \right] } \\ & { \leq \frac { 4 d L } { n } . } \end{array}
$$

Th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t ${ \mathbb E } [ h ( \sigma _ { j } ) ^ { 2 } ] \le 2 L$ (see the <sub>p</sub>roof of Theorem 3.7). A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Markov’s ine<sub>q</sub>ualit<sub>y</sub> com<sub>p</sub>letes the <sub>p</sub>roof of the claim. □

W<sub>e</sub> <sub>now</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f <sub>our</sub> <sub>ma</sub>i<sub>n</sub> l<sub>emma,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>says</sub> th<sub>a</sub>t $\operatorname* { P r } [ \mathcal T ( { \pmb x } _ { i } ) \geq d \tau / n ] = \Omega ( 1 / n )$

Proof of Theorem 3.8. Let $c > 0$ b<sub>e</sub> th<sub>e</sub> <sub>cons</sub>t<sub>an</sub>t f<sub>rom</sub> Th<sub>eorem</sub> 3<sub>.</sub>11<sub>,</sub> <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> � <sub>as</sub> th<sub>e</sub> <sub>even</sub>t th<sub>a</sub>t

$$
\mathbb { E } [ { \mathcal { T } } ( x _ { i } ) \mid X , s ] \geq { \frac { c d } { 2 n } } { \mathrm { ~ a n d ~ } } \operatorname { V a r } [ \mathbb { E } [ { \mathcal { T } } ( x _ { i } ) \mid \Sigma ] \mid X , s ] \leq { \frac { 1 6 L d } { c n } } .
$$

Let $\mu _ { \Sigma } = \operatorname { \mathbb { E } } [ { \mathcal { T } } ( x _ { i } ) \mid \Sigma , X , s ]$ <sub>, an</sub>d l<sub>e</sub>t � b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t

$$
\mu _ { \Sigma } \geq { \frac { 1 } { 2 } } \operatorname { \mathbb { E } } [ \mu _ { \Sigma } \mid X , s ] = { \frac { 1 } { 2 } } \operatorname { \mathbb { E } } _ { : { i } } [ { \mathcal { T } } ( x _ { i } ) \mid X , s ] .
$$

Sin<sub>ce co</sub>nditi<sub>o</sub>n<sub>a</sub>l <sub>o</sub>n $G$ <sub>we</sub> h<sub>ave</sub> $\mathbb { E } [ { \mathcal { T } } ( x _ { i } ) \mid X , s ] > 0$ <sub>,</sub> we can a<sub>pp</sub>l<sub>y</sub> Pale<sub>y</sub>-Z<sub>yg</sub>mund conditional on $G$ t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>a</sub>t<sub>,</sub>

$$
\begin{array} { r l } & { \mathbb { E } [ \mathbb { 1 } _ { E } \mid \boldsymbol { X } , \boldsymbol { s } ] \vert _ { G } \geq \frac { 1 } { 4 } \frac { \mathbb { E } _ { \boldsymbol { \Sigma } } [ \mu _ { \boldsymbol { \Sigma } } \mid \boldsymbol { X } , \boldsymbol { s } ] \vert _ { G } ^ { 2 } } { \mathbb { E } _ { \boldsymbol { \Sigma } } [ \mu _ { \boldsymbol { \Sigma } } ^ { 2 } \mid \boldsymbol { X } , \boldsymbol { s } ] \vert _ { G } } } \\ & { \quad \quad \quad \quad = \frac { 1 } { 4 } \frac { \mathbb { E } _ { \boldsymbol { \Sigma } } [ \mu _ { \boldsymbol { \Sigma } } \mid \boldsymbol { X } , \boldsymbol { s } ] \vert _ { G } ^ { 2 } } { \mathrm { V a r } [ \mu _ { \boldsymbol { \Sigma } } \mid \boldsymbol { X } , \boldsymbol { s } ] \vert _ { G } + \mathbb { E } _ { \boldsymbol { \Sigma } } [ \mu _ { \boldsymbol { \Sigma } } \mid \boldsymbol { X } , \boldsymbol { s } ] \vert _ { G } ^ { 2 } } } \\ & { \quad \quad \quad \quad \geq \frac { 1 } { 6 } } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>na</sub>l i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> <sub>s</sub>i<sub>nce</sub> b<sub>y</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f � <sub>an</sub>d $\mu _ { \Sigma }$ <sub>,</sub> <sub>an</sub>d <sub>our</sub> <sub>se</sub>tti<sub>ng</sub> <sub>o</sub>f $k ,$

$$
\frac { \operatorname { V a r } [ \mu _ { \Sigma } \mid X , s ] | _ { G } } { \operatorname { \mathbb { E } } _ { \Sigma } [ \mu _ { \Sigma } \mid X , s ] | _ { G } ^ { 2 } } = \frac { \operatorname { V a r } [ \mathbb { E } [ \mathcal { T } ( x _ { i } ) \mid \Sigma ] \mid X , s ] | _ { G } } { \operatorname { \mathbb { E } } _ { \Sigma , i } [ \mathcal { T } ( x _ { i } ) \mid X , s ] | _ { G } ^ { 2 } } \leq \frac { 1 6 n } { c ^ { 3 } k } \leq \frac { 1 } { 2 } .
$$

Takin<sub>g</sub> the ex<sub>p</sub>ectation over s and X, we see that $\mathbb { E } _ { X , s , \Sigma } [ \mathbb { 1 } _ { E } \mid G ] \geq 1 / 6 .$

Next we ar<sub>g</sub>ue that $\operatorname* { P r } [ G ] \geq c / 2$ <sub>.</sub> B<sub>y</sub> Th<sub>eorem</sub> 3<sub>.</sub>10<sub>,</sub> th<sub>e ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>e</sub> $\mathbb { E } [ \mathcal { T } ( \boldsymbol { x } _ { i } ) \mid \boldsymbol { X } ,$ s] can be decom<sub>p</sub>osed into a <sub>p</sub>ositive <sub>p</sub>art and a ne<sub>g</sub>ative <sub>p</sub>art. B<sub>y</sub> Theorem 3.11<sub>,</sub> the <sub>p</sub>ositive <sub>p</sub>art is at l<sub>eas</sub>t <sub>�</sub>� <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $^ { c , }$ <sub>an</sub>d b<sub>y</sub> Th<sub>eorem</sub> $3 . 1 2 ,$ the ne<sub>g</sub>ative <sub>p</sub>art is at most $4 c ^ { - 1 } \sqrt { \ell d }$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - c / 4$ <sub>, w</sub>hi<sub>c</sub>h<sub>, s</sub>i<sub>nce</sub> $\ell \leq d / C \log d$ <sub>an</sub>d $C > 0$ is lar<sub>g</sub>e enou<sub>g</sub>h<sub>,</sub> is at most $c d / 2$ If both of these events occur then the ex<sub>p</sub>ectation is at least ��/2�. Additionall<sub>y</sub>, b<sub>y</sub> Theorem 3.13, th<sub>e va</sub>ri<sub>a</sub>n<sub>ce</sub> i<sub>s a</sub>t m<sub>os</sub>t $1 6 c ^ { - 1 } L d / n$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - c / 4$ . B<sub>y</sub> the union bound � occurs <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $c / 2$

T<sub>o comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e proo</sub>f<sub>,</sub> fi<sub>x</sub> $\tau \in ( 0 , c / 4 )$ <sub>an</sub>d l<sub>e</sub>t $A _ { 1 }$ b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t $\mathcal T ( { \boldsymbol x } _ { i } ) \geq d \tau / n$ <sub>.</sub> Th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $w = w ( X , s , \Sigma ) \in [ n ]$ <sub>suc</sub>h th<sub>a</sub>t $\mathcal { T } ( \boldsymbol { x } _ { w } ) \geq \mathbb { E } _ { i } [ \mathcal { T } ( \boldsymbol { x } _ { i } ) \mid \boldsymbol { X } , \boldsymbol { s } , \Sigma ]$ . Let $A _ { 2 }$ b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t ${ i } = w$ Th<sub>en con</sub>diti<sub>ona</sub>l <sub>on</sub> � <sub>we</sub> h<sub>ave</sub> $\mathbb { 1 } _ { A _ { 1 } } \mathbb { 1 } _ { A _ { 2 } } \geq \mathbb { 1 } _ { E } \mathbb { 1 } _ { A _ { 2 } }$ . Hence,

$$
\begin{array} { r l } & { \underset { X , s , \Sigma , i } { \operatorname* { P r } } \left[ \mathcal { T } ( { \boldsymbol x } _ { i } ) \geq \frac { d \tau } { n } \right] \geq \underset { X , s , \Sigma , i } { \mathbb { E } } \left[ \mathbb { 1 } _ { A _ { 1 } } \mathbb { 1 } _ { A _ { 2 } } \right] } \\ & { \quad \quad \quad \quad \geq \frac { c } { 2 } \underset { X , s , \Sigma } { \mathbb { E } } \left[ \mathbb { 1 } _ { E } \mathbb { E } \left[ \mathbb { 1 } _ { A _ { 2 } } \mid X , s , \Sigma \right] \mid G \right] } \\ & { \quad \quad \quad \geq \frac { c } { 2 n } \underset { X , s , \Sigma } { \mathbb { E } } \left[ \mathbb { 1 } _ { E } \mid G \right] = \frac { c } { 1 2 n } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows s</sub>i<sub>nce</sub> $\operatorname* { P r } [ G ] \geq c / 2$ <sub>, an</sub>d <sub>s</sub>i<sub>nce</sub> $\mathbb { 1 } _ { E }$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $^ { i , }$ <sub>an</sub>d th<sub>e</sub> thi<sub>r</sub>d i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows s</sub>i<sub>nce</sub> $\operatorname* { P r } [ i = { \boldsymbol { w } } ] = 1 / n$ □

## 4 Lower bound for mean estimation with Mahalanobis error

In thi<sub>s sec</sub>ti<sub>o</sub>n <sub>we a</sub>r<sub>gue</sub> th<sub>a</sub>t <sub>p</sub>ri<sub>va</sub>t<sub>e</sub> m<sub>ea</sub>n <sub>es</sub>tim<sub>a</sub>ti<sub>o</sub>n in � dim<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n<sub>s w</sub>ith <sub>e</sub>rr<sub>o</sub>r r<sub>e</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> <sub>cova</sub>ri<sub>a</sub>n<sub>ce</sub> $\Sigma$ i<sub>s</sub> i<sub>mposs</sub>ibl<sub>e w</sub>ith l<sub>ess</sub> th<sub>an</sub> � <sub>samp</sub>l<sub>es.</sub> L<sub>e</sub>t $\Sigma ^ { \dagger }$ denote the <sub>p</sub>seudo-inverse of a <sub>p</sub>ositive semidefinite matrix Σ. For a vector $\boldsymbol { y } \in \mathbb { R } ^ { d }$ , the Mahalanobis norm of<sub>�</sub> relative to Σ is defined b<sub>y</sub>

$$
\begin{array} { r } { d _ { \Sigma } ( y ) = \left\{ \begin{array} { l l } { \sqrt { y ^ { \top } \Sigma ^ { \dagger } y } } & { y \in \mathrm { S p a n } ( \Sigma ) } \\ { \infty } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}
$$

O<sub>ur</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>uses</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng pr</sub>i<sub>or</sub> $\mathcal { P } _ { \mathrm { r e l } }$ over instances $( \mu , \Sigma )$

$$
( \mu , \Sigma ) \sim \mathcal { P } _ { \mathrm { r e l } } ( d ) \iff \left\{ \begin{array} { l l } { \mu \sim \mathcal { N } ( 0 , \mathbb { I } _ { d } ) } \\ { \Sigma = \mathbb { I } _ { d } - v v ^ { \top } \mathrm { ~ w h e r e ~ } v \sim \{ v : \| v \| _ { 2 } = 1 \} \mathrm { ~ u n i f o r m l y } . } \end{array} \right.\tag{17}
$$

We note that for instances � in the su<sub>pp</sub>ort of $\mathcal { P } _ { \mathrm { r e l } }$ an<sup>d</sup> an<sub>y</sub> vector $\boldsymbol { y } \in \mathbb { R } ^ { d }$ <sub>we e</sub>ith<sub>er</sub> h<sub>ave</sub> $d _ { \Sigma } ( y ) = \| y \| _ { 2 }$ if $y \in \operatorname { S p a n } ( \Sigma )$ , or $d _ { \Sigma } ( y ) = \infty$ <sub>o</sub>th<sub>erw</sub>i<sub>se.</sub>

Theorem 4.1. There exists a constant $C > 0$ such thatfor all �, $n \in \mathbb { N }$ and $\varepsilon , \delta \geq 0 , i f d \geq n + 2$ then every $( \varepsilon , \delta ) \ – D P$ mechanism $M$ satisfies

$$
\operatorname* { P r } _ { ( \mu , \Sigma ) \sim \mathcal { P } _ { r e l } } \left[ d _ { \Sigma } ( M ( X ) - \mu ) < \infty \right] \leq n \delta .
$$

Th<sub>e</sub> i<sub>n</sub>t<sub>u</sub>iti<sub>on</sub> b<sub>e</sub>hi<sub>n</sub>d th<sub>e</sub> <sub>proo</sub>f<sub>,</sub> i<sub>s</sub> th<sub>a</sub>t <sub>con</sub>diti<sub>one</sub>d <sub>on</sub> $X - \mu ,$ <sub>,</sub> th<sub>e</sub> <sub>span</sub> <sub>o</sub>f � i<sub>s</sub> <sub>compose</sub>d <sub>o</sub>f $\operatorname { S p a n } ( X - \mu )$ <sub>an</sub>d <sub>a un</sub>if<sub>orm</sub>l<sub>y ran</sub>d<sub>om</sub> $d - n - 1$ di<sub>mens</sub>i<sub>ona</sub>l <sub>su</sub>b<sub>space o</sub>f $\operatorname { S p a n } ( X - \mu ) ^ { \perp }$ <sub>.</sub> If th<sub>e</sub> al<sub>g</sub>orithm’s out<sub>p</sub>ut is not in $\operatorname { S p a n } ( X - \mu )$ <sub>,</sub> then with <sub>p</sub>robabilit<sub>y</sub> 1 over the <sub>p</sub>osterior sam<sub>p</sub>lin<sub>g</sub> of $v ,$ it<sub>s</sub> M<sub>a</sub>h<sub>a</sub>l<sub>ano</sub>bi<sub>s error</sub> i<sub>s</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>e.</sub> Th<sub>us,</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m</sub>’<sub>s ou</sub>t<sub>pu</sub>t <sub>mus</sub>t b<sub>e w</sub>ithi<sub>n</sub> $\operatorname { S p a n } ( X - \mu )$ <sub>,</sub> f<sub>rom</sub> which we derive a <sub>p</sub>rivac<sub>y</sub> violation.

Proof. We show a contradiction to $( \varepsilon , \delta ) \mathrm { - D P }$ b<sub>y</sub> constructin<sub>g</sub> an attack that distin<sub>g</sub>uishes between $z \sim \mathcal { N } ( \mu , \Sigma )$ <sub>an</sub>d $\scriptstyle x _ { i }$ <sub>, w</sub>h<sub>ere</sub> $i \sim [ n ]$ uniforml<sub>y</sub>. The test uses the out<sub>p</sub>ut of the mechanism y, as <sub>we</sub>ll <sub>as</sub> th<sub>e o</sub>th<sub>er</sub> $n - 1$ data<sub>p</sub>oints $X _ { - i } .$ <sub>.</sub> F<sub>or a</sub>ll $z , \mu \in \mathbb { R } ^ { d }$ <sub>an</sub>d d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> $X \in ( \mathbb { R } ^ { d } ) ^ { n - 1 }$ d<sub>e</sub>fi<sub>ne</sub>

$$
\begin{array} { r } { S _ { X , \mu } ( z ) = \operatorname { S p a n } \{ x _ { 1 } - \mu , \dotsc , x _ { n - 1 } - \mu , z - \mu \} \setminus \operatorname { S p a n } \{ x _ { 1 } - \mu , \dotsc , x _ { n - 1 } - \mu \} . } \end{array}
$$

A<sub>n</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> t<sub>es</sub>t $T _ { X , \mu , y } ( z ) = \mathbb { 1 } \left[ y - \mu \in S _ { X , \mu } ( z ) \right]$ <sub>.</sub> I<sub>n</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>d<sub>er</sub> <sub>o</sub>f th<sub>e</sub> <sub>proo</sub>f<sub>,</sub> <sub>we</sub> <sub>w</sub>ill <sub>om</sub>it th<sub>e</sub> subscri<sub>p</sub>ts $X , \mu$ <sub>an</sub>d $y ,$ <sub>s</sub>i<sub>nce</sub> th<sub>ey w</sub>ill b<sub>e c</sub>l<sub>ear</sub> f<sub>rom con</sub>t<sub>ex</sub>t<sub>.</sub>

Let $( \mu , \Sigma ) \sim \mathcal { P } _ { \mathrm { r e l } } , X \sim \mathcal { N } ( \mu , \Sigma ) ^ { \otimes n } , y \sim \mathcal { M } ( S ) , z \sim \mathcal { N } ( \mu , \Sigma )$ <sub>, an</sub>d $i \sim [ n ]$ <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> <sub>an</sub>d inde<sub>p</sub>endentl<sub>y</sub>. Let $G$ b<sub>e</sub> th<sub>e</sub> <sub>even</sub>t th<sub>a</sub>t $d _ { \Sigma } ( y - \mu ) < \infty$ <sub>an</sub>d l<sub>e</sub>t � b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t $y - \mu \in$ $\operatorname { S p a n } ( X - \mu )$ <sub>.</sub> Th<sub>e</sub> k<sub>ey</sub> id<sub>ea</sub> b<sub>e</sub>hi<sub>n</sub>d <sub>our</sub> <sub>proo</sub>f i<sub>s</sub> th<sub>a</sub>t $\operatorname* { P r } [ G ] = \operatorname* { P r } [ E ]$ <sub>,</sub> th<sub>a</sub>t i<sub>s,</sub> th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t th<sub>e</sub> M<sub>a</sub>h<sub>a</sub>l<sub>ano</sub>bi<sub>s</sub> <sub>error</sub> i<sub>s</sub> fi<sub>n</sub>it<sub>e</sub> i<sub>s</sub> <sub>exac</sub>tl<sub>y</sub> th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t ${ \bf { \nabla } } _ { \bf { \boldsymbol { y } } } - { \boldsymbol { \mu } } _ { \bf { \boldsymbol { \mu } } }$ is in the s<sub>p</sub>an of $X - \mu$ <sub>.</sub> T<sub>o</sub> <sub>see</sub> <sub>w</sub>h<sub>y</sub> thi<sub>s</sub> i<sub>s, o</sub>b<sub>serve</sub> th<sub>a</sub>t

$$
\begin{array} { r l } & { \mathbb { E } [ \mathbb { 1 } _ { G } ] = \mathbb { E } \big [ \mathbb { 1 } _ { G } \mathbb { 1 } _ { E } \big ] + \mathbb { E } \big [ \mathbb { 1 } _ { G } \mathbb { 1 } _ { \overline { { E } } } \big ] } \\ & { \qquad = \mathbb { E } \big [ \mathbb { 1 } _ { E } \big ] + \underset { X , y , \mu } { \mathbb { E } } \big [ \mathbb { E } \big [ \mathbb { 1 } _ { G } \mid X , y , \mu \big ] \mathbb { 1 } _ { \overline { { E } } } \big ] } \\ & { \qquad = \mathbb { E } \big [ \mathbb { 1 } _ { E } \big ] } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>secon</sub>d <sub>equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> <sub>s</sub>i<sub>nce</sub> 1 $\lfloor _ { E } = 1$ im<sub>p</sub>lies $\mathbb { 1 } _ { G } = 1$ <sub>.</sub> Th<sub>e</sub> fi<sub>na</sub>l <sub>equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> <sub>s</sub>i<sub>nce</sub> <sub>con</sub>diti<sub>ona</sub>l <sub>on</sub> $y , X$ <sub>, an</sub>d $\mu$ <sub>suc</sub>h th<sub>a</sub>t $\mathbb { 1 } _ { \overline { { E } } } = 1$ , over the random sam<sub>p</sub>lin<sub>g</sub> of the direction v (see (17)), <sub>we</sub> h<sub>ave</sub> $\langle v , y - \mu \rangle \neq 0$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> 1<sub>,</sub> <sub>an</sub>d th<sub>us</sub> $d _ { \Sigma } ( y - \mu ) = \infty$ , <sup>i</sup>m<sub>p</sub><sup>l</sup><sub>y</sub><sup>i</sup>n<sub>g</sub> $\operatorname { \mathbb { E } } _ { v } [ \mathbb { 1 } _ { G } \mid X , y , \mu ] = 0$

IN case. Since $n \ <$ � th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> $\{ x _ { i } - \mu \} _ { i \in [ n ] }$ <sub>are</sub> li<sub>near</sub>l<sub>y</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> 1<sub>,</sub> <sub>an</sub>d th<sub>us</sub> $\bigcap { } _ { i \in [ n ] }$ <sup>S</sup><sub>p</sub>an $( X _ { - i } - \mu ) = \{ 0 \}$ <sub>.</sub> It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t th<sub>ere ex</sub>i<sub>s</sub>t<sub>s some ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>e</sub> $w \in [ n ]$ <sub>suc</sub>h th<sub>a</sub>t <sub>w</sub>h<sub>en</sub> $y \ne \mu$ <sub>an</sub>d <sub>con</sub>diti<sub>ona</sub>l <sub>on</sub> �<sub>, we</sub> h<sub>ave</sub> $y - \mu \not \in \mathrm { S p a n } ( X _ { - w } - \mu )$ <sub>,</sub> or e<sub>q</sub>uivalentl<sub>y</sub> $T _ { X _ { - w } , \mu } ( x _ { w } ) = 1$ <sub>,</sub> with <sub>p</sub>robabilit<sub>y</sub> 1. Since ${ \bf { \nabla } } _ { i } = w$ with <sub>p</sub>robabilit<sub>y</sub> $1 / n$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>o</sub>f $y , X$ , u <sub>an</sub>d �<sub>, we o</sub>bt<sub>a</sub>i<sub>n,</sub>

$$
\begin{array} { r l } & { \operatorname* { P r } \bigl [ T _ { X _ { - i } , \mu } ( x _ { i } ) = 1 \bigr ] \geq \displaystyle \frac { 1 } { n } \operatorname* { P r } [ T _ { X _ { - w } , \mu } ( x _ { w } ) = 1 \mid E \cap \{ y \neq \mu \} ] \operatorname* { P r } [ E \cap \{ y \neq \mu \} ] } \\ & { \qquad \geq \displaystyle \frac { \operatorname* { P r } [ E ] - \operatorname* { P r } [ y = \mu ] } { n } = \displaystyle \frac { \operatorname* { P r } [ G ] } { n } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>equa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s s</sub>i<sub>nce</sub> $\operatorname* { P r } [ y = \mu ] = 0$ <sub>an</sub>d $\operatorname* { P r } [ E ] = \operatorname* { P r } [ G ]$

OUT case. Since $n \ \leq \ d - 2 ,$ <sub>,</sub> f<sub>or</sub> fi<sub>xe</sub>d $y , \mu , X$ <sub>we</sub> h<sub>ave</sub> $z - \mu$ ∉ S<sub>p</sub>an $\{ X - \mu , y - \mu \}$ <sub>w</sub>ith <sub>p</sub>robabilit<sub>y</sub> 1 over the randomness of z. Thus, $\mathrm { P r } [ T _ { X _ { - i } , \mu } ( z ) = 1 ] = 0$

Bounding the accuracy. We use the definition of $( \varepsilon , \delta ) \mathrm { - D P }$ to bound the accurac<sub>y</sub> of M.

$$
\begin{array} { r l } { \displaystyle \frac { \operatorname* { P r } [ G ] } { n } \le \operatorname* { P r } \left[ T _ { X _ { - i } , \mu } ( x _ { i } ) = 1 \right] } \\ { \le e ^ { \varepsilon } \operatorname* { P r } [ T _ { X _ { - i } , \mu } ( x _ { i } ) = 1 ] + \delta } \quad \quad \quad  & { \mathrm { u s i n g ~ } T ( z ) \approx _ { ( \varepsilon , \delta ) } \ T ( x _ { i } ) } \\ { = \delta . } \end{array}
$$

Rearran<sub>g</sub>in<sub>g</sub> terms<sub>,</sub> we see that $\operatorname* { P r } [ d _ { \Sigma } ( { \pmb y } - { \pmb \mu } ) < \infty ] \leq n \delta$

## References

[AAK21] I<sub>s</sub>h<sub>aq</sub> Ad<sub>en-</sub>Ali<sub>,</sub> H<sub>assan</sub> A<sub>s</sub>hti<sub>an</sub>i<sub>,</sub> <sub>an</sub>d G<sub>au</sub>t<sub>am</sub> K<sub>ama</sub>th<sub>.</sub> “O<sub>n</sub> th<sub>e</sub> S<sub>amp</sub>l<sub>e</sub> C<sub>omp</sub>l<sub>ex</sub>it<sub>y</sub> <sub>o</sub>f Privately Learning Unbounded High-Dimensional Gaussians.” International Conference on Algorithmic Learning Theory. ALT 2021.

[ALNP24] Martin Aumüller, Christian Janos Lebeda, Boel Nelson, and Rasmus Pa<sub>g</sub>h. “PLAN: Variance-Aware Private Mean Estimation.” Privacy Enhancing Technologies Symposium. <sup>PETS</sup> 2024. (Link).

[BBFST21] Gavin Brown, Mark Bun, Vital<sub>y</sub> Feldman, Adam Smith, and Kunal Talwar. “When is M<sub>e</sub>m<sub>o</sub>riz<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f Irr<sub>e</sub>l<sub>eva</sub>nt Tr<sub>a</sub>inin<sub>g</sub> D<sub>a</sub>t<sub>a</sub> N<sub>ecessa</sub>r<sub>y</sub> f<sub>o</sub>r Hi<sub>g</sub>h-A<sub>ccu</sub>r<sub>acy</sub> L<sub>ea</sub>rnin<sub>g</sub>?” arXiv preprint arXiv:2012.06421 (2021). (Link).

[BGSUZ21] Gavin Brown, Marco Gaboardi, Adam Smith, Jonathan Ullman, and L<sub>y</sub>dia Zak<sub>y</sub>nthinou. “C<sub>ovar</sub>i<sub>ance-</sub>A<sub>ware</sub> P<sub>r</sub>i<sub>va</sub>t<sub>e</sub> M<sub>ean</sub> E<sub>s</sub>ti<sub>ma</sub>ti<sub>on</sub> With<sub>ou</sub>t P<sub>r</sub>i<sub>va</sub>t<sub>e</sub> C<sub>ovar</sub>i<sub>ance</sub> E<sub>s</sub>ti<sub>ma</sub>ti<sub>on.</sub>” Advances in Neural Information Processing Systems. NeurIPS 2021. (Link).

[BHS23] G<sub>av</sub>i<sub>n</sub> B<sub>rown,</sub> S<sub>amue</sub>l B<sub>.</sub> H<sub>op</sub>ki<sub>ns,</sub> <sub>an</sub>d Ad<sub>am</sub> S<sub>m</sub>ith<sub>.</sub> “F<sub>as</sub>t<sub>,</sub> S<sub>amp</sub>l<sub>e-</sub>Efi<sub>c</sub>i<sub>en</sub>t<sub>,</sub> Afi<sub>ne-</sub> In<sub>va</sub>ri<sub>a</sub>nt Pri<sub>va</sub>t<sub>e</sub> M<sub>ea</sub>n <sub>a</sub>nd C<sub>ova</sub>ri<sub>a</sub>n<sub>ce</sub> E<sub>s</sub>tim<sub>a</sub>ti<sub>o</sub>n f<sub>o</sub>r S<sub>u</sub>b<sub>gauss</sub>i<sub>a</sub>n Di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub>” Conference on Learning Theory. COLT 2023.

[BKSW19] Mark Bun, Gautam Kamath, Thomas Steinke, and Zhiwei Steven Wu. “Private H<sub>yp</sub>othesis Selection.” Advances in Neural Information Processing Systems. NeurIPS 2019.

[BUV14] Mark Bun, Jonathan Ullman, and Salil Vadhan. “Fin<sub>g</sub>er<sub>p</sub>rintin<sub>g</sub> Codes and the Price of Approximate Diferential Privacy.” ACM Symposium on Theory of Computing. STOC 2014. (Link).

[Car+21] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Ja<sub>g</sub>ielski, Ariel Herbert-Voss, K<sub>a</sub>th<sub>er</sub>i<sub>ne</sub> L<sub>ee,</sub> Ad<sub>am</sub> R<sub>o</sub>b<sub>er</sub>t<sub>s,</sub> T<sub>om</sub> B<sub>rown,</sub> <sub>e</sub>t <sub>a</sub>l<sub>.</sub> “E<sub>x</sub>t<sub>rac</sub>ti<sub>ng</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a</sub> f<sub>rom</sub> l<sub>arge</sub> language models.” USENIX Security Symposium. USENIX Security 2021. (Link).

[CDMM26] Sitan Chen, Jin<sub>gq</sub>iu Din<sub>g</sub>, Mahbod Majid, and Walter McKelvie. “Com<sub>p</sub>utation-Utilit<sub>y</sub>- Privacy Tradeofs in Bayesian Estimation.” ACM Symposium on Theory of Computing. <sup>STOC</sup> 2026. (Link).

[CLEKS19] Nicholas Carlini, Chan<sub>g</sub> Liu, <sup>Ú</sup>lfar Erlin<sub>g</sub>sson, Jernej Kos, and Dawn Son<sub>g</sub>. “The secret sharer: Evaluating and testing unintended memorization in neural networks.” USENIX Security Symposium. USENIX Security 2019. (Link).

[CWZ19] T. Ton<sub>y</sub> Cai, Yichen Wan<sub>g</sub>, and Linjun Zhan<sub>g</sub>. “The Cost of Privac<sub>y</sub>: O<sub>p</sub>timal Rates of Convergence for Parameter Estimation with Diferential Privacy.” arXiv preprint arXiv:1902.04495 (2019). (Link).

[CWZ23] T. Ton<sub>y</sub> Cai, Yichen Wan<sub>g</sub>, and Linjun Zhan<sub>g</sub>. “Score attack: A lower bound technique for optimal diferentially private learning.” arXiv preprint arXiv:2303.07152 (2023). (Link).

[DJYZZ24] Yuval Da<sub>g</sub>an, Michael Jordan, Xuelin Yan<sub>g</sub>, L<sub>y</sub>dia Zak<sub>y</sub>nthinou, and Nikita Zhivotovski<sub>y</sub>. “Dimension-free private mean estimation for anisotropic distributions.” Advances in Neural Information Processing Systems. NeurIPS 2024. (Link).

[DL22] Jules De<sub>p</sub>ersin and Guillaume Lecué. “O<sub>p</sub>timal robust mean and location estimation via convex programs with respect to any pseudo-norms.” Probability Theory and Related Fields 183.3 (2022).

[DMNS06] C<sub>y</sub>nthia Dwork, Frank McSherr<sub>y</sub>, Kobbi Nissim, and Adam Smith. “Calibratin<sub>g</sub> Noise to Sensitivity in Private Data Analysis.” Conference on Theory of Cryptography. TCC 2006. (Link).

[DMT07] C<sub>y</sub>nthia Dwork, Frank McSherr<sub>y</sub>, and Kunal Talwar. “The <sub>p</sub>rice of <sub>p</sub>rivac<sub>y</sub> and the limits of LP decoding.” ACM Symposium on Theory of Computing. STOC 2007. (Link).

[DN03] Irit Dinur and Kobbi Nissim. “Revealing information while preserving privacy.” ACM Symposium on Principles ofDatabase Systems. PODS 2003. (Link).

[DSSUV15] C<sub>y</sub>nthia Dwork, Adam Smith, Thomas Steinke, Jonathan Ullman, and Salil Vadhan. “Robust Traceability from Trace Amounts.” IEEE Symposium on Foundations of Computer Science. FOCS 2015. (Link).

[DY08] C<sub>yn</sub>thi<sub>a</sub> D<sub>wor</sub>k <sub>an</sub>d S<sub>ergey</sub> Y<sub>e</sub>kh<sub>an</sub>i<sub>n.</sub> “N<sub>ew</sub> <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>a</sub>tt<sub>ac</sub>k<sub>s</sub> <sub>on</sub> <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l di<sub>sc</sub>l<sub>osure</sub> control mechanisms.” International Cryptology Conference. CRYPTO 2008. (Link).

[Fel20] Vit<sub>a</sub>l<sub>y</sub> F<sub>e</sub>ld<sub>man.</sub> “D<sub>oes</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>requ</sub>i<sub>re</sub> <sub>memor</sub>i<sub>za</sub>ti<sub>on</sub>? A <sub>s</sub>h<sub>or</sub>t t<sub>a</sub>l<sub>e</sub> <sub>a</sub>b<sub>ou</sub>t <sub>a</sub> l<sub>ong</sub> t<sub>a</sub>il<sub>.</sub>” ACM Symposium on Theory of Computing. STOC 2020. (Link).

[GLSUZ26] Renaud Gaucher, E<sub>p</sub>hraim Linder, Adam Smith, Jonathan Ullman, and L<sub>y</sub>dia Zak<sub>y</sub>nthinou. A Simple Presentation of Robust Membership Inference Attacks. 2026. arXiv: 2610.36458 [cs.CR]. (Link).

[Hom+08] Nils Homer, Szabolcs Szelin<sub>g</sub>er, Mar<sub>g</sub>ot Redman, David Du<sub>gg</sub>an, Waibhav Tembe, Jill Muehlin<sub>g,</sub> John V. Pearson<sub>,</sub> Dietrich A. Ste<sub>p</sub>han<sub>,</sub> et al. “Resolvin<sub>g</sub> individuals contributin<sub>g</sub> trace amounts of DNA to hi<sub>g</sub>hl<sub>y</sub> com<sub>p</sub>lex mixtures usin<sub>g</sub> hi<sub>g</sub>h-densit<sub>y</sub> SNP genotyping microarrays.” PLoS Genetics 4.8 (2008). (Link).

[KDH23] Rohith Kuditi<sub>p</sub>udi, John C. Duchi, and Saminul Ha<sub>q</sub>ue. “A Prett<sub>y</sub> Fast Al<sub>g</sub>orithm for Adaptive Private Mean Estimation.” Conference on Learning Theory. COLT 2023.

[KLSU19] Gautam Kamath, Jerr<sub>y</sub> Li, Vikrant Sin<sub>g</sub>hal, and Jonathan Ullman. “Privatel<sub>y</sub> Learnin<sub>g</sub> High Dimensional Distributions.” Conference on Learning Theory. COLT 2019. (Link).

[KMS22] Gautam Kamath, Ar<sub>gy</sub>ris Mouzakis, and Vikrant Sin<sub>g</sub>hal. “New Lower Bounds for Private Estimation and a Generalized Fingerprinting Lemma.” Advances in Neural Information Processing Systems. NeurIPS 2022.

[KPST26] S<sub>y</sub>amantak Kumar, Shour<sub>y</sub>a Pande<sub>y</sub>, Purnamrita Sarkar, and Kevin Tian. “On the Curse of Dimensionality in Private Sparse Covariance Estimation and PCA.” Conference on Learning Theory. COLT 2026. (Link).

[KRSU10] Shiva Prasad Kasiviswanathan, Mark Rudelson, Adam Smith, and Jonathan Ullman. “Th<sub>e pr</sub>i<sub>ce o</sub>f <sub>pr</sub>i<sub>va</sub>t<sub>e</sub>l<sub>y re</sub>l<sub>eas</sub>i<sub>ng con</sub>ti<sub>ngency</sub> t<sub>a</sub>bl<sub>es an</sub>d th<sub>e spec</sub>t<sub>ra o</sub>f <sub>ran</sub>d<sub>om ma</sub>t<sub>r</sub>i<sub>ces</sub> with correlated rows.” ACM Symposium on Theory of Computing. STOC 2010. (Link).

[KV18] Vi<sub>s</sub>h<sub>es</sub>h K<sub>arwa</sub> <sub>an</sub>d S<sub>a</sub>lil V<sub>a</sub>dh<sub>an.</sub> “Fi<sub>n</sub>it<sub>e</sub> S<sub>amp</sub>l<sub>e</sub> Dif<sub>eren</sub>ti<sub>a</sub>ll<sub>y</sub> P<sub>r</sub>i<sub>va</sub>t<sub>e</sub> C<sub>on</sub>fid<sub>ence</sub> I<sub>n-</sub> tervals.” Conference on Innovations in Theoretical Computer Science. ITCS 2018. (Link).

[LKKO21] Xi<sub>y</sub>an<sub>g</sub> Liu, Weihao Kon<sub>g</sub>, Sham Kakade, and Sewoon<sub>g</sub> Oh. “Robust and diferentiall<sub>y</sub> private mean estimation.” Advances in Neural Information Processing Systems. NeurIPS 2021.

[LM00] Bé<sub>a</sub>t<sub>r</sub>i<sub>ce</sub> L<sub>auren</sub>t <sub>an</sub>d P<sub>asca</sub>l M<sub>assar</sub>t<sub>.</sub> “Ad<sub>ap</sub>ti<sub>ve es</sub>ti<sub>ma</sub>ti<sub>on o</sub>f <sub>a qua</sub>d<sub>ra</sub>ti<sub>c</sub> f<sub>unc</sub>ti<sub>ona</sub>l b<sub>y</sub> model selection.” The Annals ofStatistics 28.5 (2000). (Link).

[MZ25] A<sub>rs</sub>h<sub>a</sub>k Mi<sub>nasyan</sub> <sub>an</sub>d Nikit<sub>a</sub> Zhi<sub>vo</sub>t<sub>ovs</sub>ki<sub>y.</sub> “St<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> O<sub>p</sub>ti<sub>ma</sub>l R<sub>o</sub>b<sub>us</sub>t M<sub>ean</sub> <sub>an</sub>d Covariance Estimation for Anisotropic Gaussians.” Mathematical Statistics and Learning. Vo<sup>l</sup>. 8. 2025. (Link).

[Nar25] Sh<sub>yam</sub> N<sub>arayanan.</sub> “B<sub>e</sub>tt<sub>er</sub> <sub>an</sub>d Si<sub>mp</sub>l<sub>er</sub> L<sub>ower</sub> B<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> Dif<sub>eren</sub>ti<sub>a</sub>ll<sub>y</sub> P<sub>r</sub>i<sub>va</sub>t<sub>e</sub> St<sub>a-</sub> tistical Estimation.” IEEE Trans. Inf. Theor. 71.2 (Feb. 2025). issn: 0018-9448. doi: 10. 1109/TIT.2024.3511624. (Link).

[PH25] Victor S. Portella and Nicholas J. A. Harve<sub>y</sub>. “Lower Bounds for Private Estimation of Gaussian Covariance Matrices under All Reasonable Parameter Regimes.” Conference on Learning Theory. COLT 2025. (Link).

[PTU24] Nat<sub>y</sub> Peter<sub>,</sub> Eliad Tsfadia<sub>,</sub> and Jonathan Ullman. “Smooth Lower Bounds for Diferentially Private Algorithms via Padding-and-Permuting Fingerprinting Codes.” Conference on Learning Theory. COLT 2024. (Link).

[Smi11] Adam Smith. “Privac<sub>y</sub>-Preservin<sub>g</sub> Statistical Estimation with O<sub>p</sub>timal Conver<sub>g</sub>ence Rates.” ACM Symposium on Theory of Computing. STOC 2011.

[SOJH09] Sriram Sankararaman, Guillaume Obozinski, Michael I. Jordan, and Eran Hal<sub>p</sub>erin. “Genomic privacy and limits of individual detection in a pool.” Nature Genetics 41.9 (2009). (Link).

[SS21] Vikrant Singhal and Thomas Steinke. “Privately Learning Subspaces.” Advances in Neural Information Processing Systems. NeurIPS 2021.

[SSSS17] R<sub>eza</sub> Sh<sub>o</sub>k<sub>r</sub>i<sub>,</sub> M<sub>arco</sub> St<sub>rona</sub>ti<sub>,</sub> C<sub>ongz</sub>h<sub>eng</sub> S<sub>ong,</sub> <sub>an</sub>d Vit<sub>a</sub>l<sub>y</sub> Sh<sub>ma</sub>tik<sub>ov.</sub> “M<sub>em</sub>b<sub>ers</sub>hi<sub>p</sub> Inference Attacks against Machine Learning Models.” IEEE Symposium on Security & Privacy. IEEE S&P 2017. (Link).

[SU17] Thomas Steinke and Jonathan Ullman. “Ti<sub>g</sub>ht Lower Bounds for Diferentiall<sub>y</sub> Private Selection.” IEEE Symposium on Foundations of Computer Science. FOCS 2017.

[TCKMS22] Eliad Tsfadia, Edith Cohen, Haim Ka<sub>p</sub>lan, Yisha<sub>y</sub> Mansour, and Uri Stemmer. “Friendl<sub>y</sub>- Core: Practical Diferentially Private Aggregation.” International Conference on Machine Learning. ICML 2022. (Link).

[Tsf24] Eli<sub>a</sub>d T<sub>s</sub>f<sub>a</sub>di<sub>a.</sub> “O<sub>n</sub> dif<sub>eren</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>pr</sub>i<sub>va</sub>t<sub>e</sub> <sub>su</sub>b<sub>space</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> i<sub>n</sub> <sub>a</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on-</sub>f<sub>ree</sub> setting.” Advances in Neural Information Processing Systems. NeurIPS 2024.

[Ver26] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. 2nd ed. Cambridge Series in Statistical and Probabilistic Mathematics. C<sub>a</sub>mbrid<sub>ge</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> Pr<sub>ess,</sub> 2026<sub>.</sub>

[YGFJ18] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. “Privac<sub>y</sub> risk in machine learning: Analyzing the connection to overfitting.” IEEE Computer Security Foundations Symposium. CSF 2018. (Link).

## A Concentration Results

Lemma A.1 (Concentration of the empirical covariance). Let Σ be a rank-� orthogonal projection in $\mathbb { R } ^ { d }$ , and $\boldsymbol { X } \sim \boldsymbol { N } ( 0 , \boldsymbol { \Sigma } ) ^ { \otimes n }$ . There is a numerical constant $c > 0$ , such that, $i f n \leq c k .$ , then

$$
\frac { k } { 2 n } P _ { X } \preceq \hat { \Sigma } _ { X } \preceq \frac { 2 k } { n } P _ { X }
$$

with probability at least $1 - 2 \exp ( - c k )$

Proof of Theorem A.1. The result is a direct consequence of standard results on the concentration of ei<sub>g</sub>envalues of matrices with i.i.d. standard <sub>g</sub>aussian entries. Indeed<sub>,</sub> the em<sub>p</sub>irical covariance matrix and the <sub>p</sub>rojector on S<sub>p</sub>an(X) write (for $n \leq k \leq d )$

$$
{ \hat { \Sigma } } _ { X } = { \frac { 1 } { n } } X X ^ { \top } = \sum _ { i \in [ n ] } { \frac { \sigma _ { i } ^ { 2 } ( X ) } { n } } d _ { i } ( X ) d _ { i } ( X ) ^ { \top } { \mathrm { ~ a n d ~ } } P _ { X } = \sum _ { i \in [ n ] } d _ { i } ( X ) d _ { i } ( X ) ^ { \top } ,
$$

<sub>w</sub>ith $( d _ { i } ( \pmb { X } ) ) _ { i \in [ n ] } \in \mathbb { R } ^ { d }$ <sub>an or</sub>th<sub>onorma</sub>l f<sub>am</sub>il<sub>y, an</sub>d $( \sigma _ { i } ^ { 2 } ( X ) ) _ { i \in [ n ] }$ th<sub>e s</sub>i<sub>ngu</sub>l<sub>ar va</sub>l<sub>ues o</sub>f $X X ^ { \top }$ (and also of $X ^ { \top } X )$ . Since<sub>,</sub> in the s<sub>p</sub>an of $\Sigma , X$ i<sub>s a</sub> $k \times n$ matrix with i.i.d. standard <sub>g</sub>a<sub>u</sub>ssian entries, Theorem 4.6.1 in [Ver26] states that, with <sub>p</sub>robabilit<sub>y</sub> at least $1 - \beta ,$

$$
\left\| { \frac { 1 } { k } } X ^ { \mathsf { T } } X - \mathbb { I } _ { n } \right\| _ { 2 } \leq C \operatorname* { m a x } { \big ( } a , a ^ { 2 } { \big ) } \quad { \mathrm { w i t h } } \quad a = { \sqrt { \frac { n } { k } } } + { \sqrt { \frac { \log ( 2 / \beta ) } { k } } } ,
$$

<sub>w</sub>ith $C > 0$ <sub>a un</sub>i<sub>versa</sub>l <sub>cons</sub>t<sub>an</sub>t<sub>.</sub> It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t

$$
\operatorname* { m a x } ( 0 , 1 - C \operatorname* { m a x } ( a , a ^ { 2 } ) ) P _ { X } \preceq \frac { n } { k } \hat { \Sigma } _ { X } \preceq ( 1 + C \operatorname* { m a x } ( a , a ^ { 2 } ) ) P _ { X } .
$$

A f<sub>ew a</sub>l<sub>ge</sub>b<sub>ra</sub>i<sub>c man</sub>i<sub>pu</sub>l<sub>a</sub>ti<sub>ons s</sub>h<sub>ow</sub> th<sub>a</sub>t f<sub>or</sub> $\beta = 2 \exp ( - k / ( 4 C ) ^ { 2 } )$ <sub>an</sub>d $k > ( 4 C ) ^ { 2 } n$ it h<sub>o</sub>ld<sub>s</sub> th<sub>a</sub>t � m<sub>a</sub>x $( a , a ^ { 2 } ) \leq 1 / 2$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>conc</sub>l<sub>u</sub>d<sub>es</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub> □

Lemma A.2 (Chernof for bounded random variables). Let $r _ { 1 } , \ldots , r _ { m }$ be independent random variables taking values in $[ 0 , R ]$ . Then for all $\beta > 0$

$$
\operatorname* { P r } \left[ \sum r _ { j } \ge e \sum \mathbb { E } [ r _ { j } ] + R \ln ( 1 / \beta ) \right] \le \beta .
$$

Proof. Observe that for any � independent random variables $r _ { 1 } , \ldots , r _ { m }$ takin<sub>g</sub> values in $[ 0 , R ]$ we h<sub>ave</sub>

$$
\mathbb { E } \left[ e ^ { \sum r _ { j } / R } \right] = \Big [ \ \Big \rvert \mathbb { E } \left[ e ^ { r _ { j } / R } \right] \le \Big ] \ \mathbb { E } \left[ 1 + e r _ { j } / R \right] \le e ^ { ( e / R ) \sum \mathbb { E } \left[ r _ { j } \right] } ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t f<sub>or</sub> <sub>a</sub>ll $t ~ \in ~ \left[ 0 , 1 \right]$ <sub>we</sub> h<sub>ave</sub> $e ^ { t } ~ = ~ 1 +$ $\textstyle t \sum _ { k \geq 1 } t ^ { k - 1 } / k ! \leq 1 + e t$ <sub>.</sub> B<sub>y a</sub> Ch<sub>erno</sub>f b<sub>oun</sub>d<sub>,</sub> f<sub>or a</sub>ll $t > 0$

$$
\mathrm { P r } \left[ \sum r _ { j } / R \geq t \right] \leq e ^ { ( e / R ) \sum \mathbb { E } [ \pmb { r } _ { j } ] - t } .
$$

Settin<sub>g</sub> $t = ( e / R ) \sum \mathbb { E } [ r _ { j } ] + \ln ( 1 / \beta )$ and multi<sub>p</sub>l<sub>y</sub>in<sub>g</sub> throu<sub>g</sub>h b<sub>y</sub> � com<sub>p</sub>letes the <sub>p</sub>roof. □

Lemma A.3 (Sum of heavy tailed random variables). Let $\alpha \in ( 0 , 1 ) , t _ { 0 } > 1 / 2$ . Let y be a nonnegative random variable such that $\operatorname* { P r } [ y \geq t ] \leq C t ^ { - \alpha }$ for all $t \geq t _ { 0 }$ . Let $y _ { 1 } , \ldots , y _ { d }$ be i.i.d. copies ofy. Then there exists $K = K ( C , \alpha , t _ { 0 } )$ , such that for all $m \geq 1$

$$
\operatorname* { P r } \left[ \sum _ { i = 1 } ^ { d } \pmb { y } _ { i } \ge m d ^ { 1 / \alpha } \right] \le K m ^ { - \alpha } .
$$

Additionally for $\alpha = 1$ , it holds that

$$
\operatorname* { P r } \left[ \sum _ { i = 1 } ^ { d } \pmb { y } _ { i } \geq m d \right] \leq \frac { K _ { 1 } \log ( m d ) } { m } ,
$$

where $K _ { 1 } = \left( 2 C + t _ { 0 } \right)$

Proof. Let $A : = \operatorname* { m a x } \{ C , t _ { 0 } ^ { \alpha } \}$ <sub>.</sub> Th<sub>en</sub> f<sub>or</sub> <sub>a</sub>ll $t > 0 , \operatorname* { P r } [ y \ge t ] \le \operatorname* { m i n } ( 1 , A t ^ { - \alpha } )$ . Let

$$
S = \sum _ { i = 1 } ^ { d } y _ { i } , \qquad M = \operatorname* { m a x } _ { 1 \leq i \leq d } y _ { i } , \qquad x = m d ^ { 1 / \alpha } .
$$

B<sub>y</sub> th<sub>e</sub> <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d<sub>,</sub>

$$
\operatorname* { P r } \left[ M \geq { \frac { x } { 2 } } \right] \leq d \operatorname* { P r } \left[ y \geq { \frac { x } { 2 } } \right] \leq d \operatorname* { m i n } \left( 1 , C { \left( { \frac { x } { 2 } } \right) } ^ { - \alpha } \right) = \operatorname* { m i n } ( d , A 2 ^ { \alpha } m ^ { - \alpha } ) \leq A 2 ^ { \alpha } m ^ { - \alpha } .
$$

N<sub>ow</sub> l<sub>e</sub>t $\begin{array} { r } { \pmb { T } = \sum _ { i = 1 } ^ { d } \pmb { y _ { i } } \mathbb { 1 } \left[ \pmb { y _ { i } } \le x / 2 \right] . \mathrm { I f } \pmb { S } \ge x \mathrm { ~ a n d ~ } \pmb { M } < x / 2 , } \end{array}$ th<sub>en</sub> $T = S \geq x ,$ , so

$$
\operatorname* { P r } \left[ S \geq x \right] \leq \operatorname* { P r } \left( S \geq x , ~ M < { \frac { x } { 2 } } \right) + \operatorname* { P r } \left( S \geq x , ~ M \geq { \frac { x } { 2 } } \right) \leq \operatorname* { P r } [ T \geq x ] + \operatorname* { P r } [ M \geq { \frac { x } { 2 } } ] .
$$

M<sub>oreover,</sub> f<sub>or</sub> $\alpha < 1$

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \pmb { T } ] = d \mathbb { E } \left[ \pmb { y } \mathbb { 1 } \left[ \pmb { y } \leq x / 2 \right] \right] \leq d \int _ { 0 } ^ { x / 2 } \operatorname* { P r } [ \pmb { y } \geq t ] d t } \\ { \displaystyle \leq d t _ { 0 } + d A \int _ { t _ { 0 } } ^ { x / 2 } t ^ { - \alpha } d t = d t _ { 0 } + \frac { d A } { 1 - \alpha } \bigg ( \left( \frac { x } { 2 } \right) ^ { 1 - \alpha } - t _ { 0 } ^ { 1 - \alpha } \bigg ) , } \end{array}
$$

th<sub>us</sub> $\begin{array} { r } { \mathbb { E } [ T ] \leq \frac { d A } { 1 - \alpha } 2 ^ { \alpha - 1 } x ^ { 1 - \alpha } } \end{array}$ . B<sub>y</sub> Markov’s ine<sub>q</sub>ualit<sub>y,</sub>

$$
\operatorname* { P r } [ T \geq x ] \leq \frac { \mathbb { E } [ T ] } { x } \leq \frac { A 2 ^ { \alpha - 1 } } { 1 - \alpha } d x ^ { - \alpha } \leq \frac { A 2 ^ { \alpha - 1 } } { 1 - \alpha } m ^ { - \alpha } .
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> b<sub>y</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $x = m d ^ { 1 / \alpha }$ , so $d x ^ { - \alpha } = m ^ { - \alpha }$ . Combinin<sub>g</sub> the two:

$$
\operatorname* { P r } [ S \geq m d ^ { 1 / \alpha } ] \leq A 2 ^ { \alpha } m ^ { - \alpha } + \frac { A 2 ^ { \alpha - 1 } } { 1 - \alpha } m ^ { - \alpha } \leq K _ { \alpha , t _ { 0 } , C } m ^ { - \alpha }
$$

<sub>w</sub>h<sub>ere</sub> $\begin{array} { r } { K : = \operatorname* { m a x } \{ C , t _ { 0 } ^ { \alpha } \} \Bigl ( 2 ^ { \alpha } + \frac { 2 ^ { \alpha - 1 } } { 1 - \alpha } \Bigr ) } \end{array}$

F<sub>o</sub>r $\alpha = 1 ,$

$$
\mathbb { E } [ T ] \le d t _ { 0 } + d C \int _ { t _ { 0 } } ^ { x / 2 } t ^ { - \alpha } d t = d t _ { 0 } + d C \log { \left( \frac { x } { 2 t _ { 0 } } \right) } .
$$

Th<sub>us</sub> $\begin{array} { r } { \mathrm { P r } [ { \cal T } \ge x ] \le \frac { t _ { 0 } } { m } + \frac { C \log \left( m d / 2 t _ { 0 } \right) } { m } } \end{array}$ A<sub>n</sub>d

$$
\operatorname* { P r } [ S \geq m d ^ { 1 / \alpha } ] \leq { \frac { 2 C } { m } } + { \frac { t _ { 0 } } { m } } + { \frac { C \log ( m d / 2 t _ { 0 } ) } { m } } \leq K _ { 1 } { \frac { \log ( m d ) } { m } }
$$

<sub>w</sub>h<sub>ere</sub> $K _ { 1 } = \left( 3 C + t _ { 0 } \right)$

Lemma A.4 (Top-� samples from inv $\chi ^ { 2 } ( 1 , 1 ) )$ . Let $\pmb { \sigma } _ { 1 } ^ { 2 } , . . . , \pmb { \sigma } _ { d } ^ { 2 } \sim \mathrm { i n v } \chi ^ { 2 } ( 1 , 1 ) ^ { \otimes d }$ , then there exists an absolute constant $c > 0$ such that or all $1 \leq k \leq c d$

$$
\operatorname* { P r } \left[ \pmb { \sigma } _ { ( k ) } ^ { 2 } \geq \frac { d ^ { 2 } } { 4 k ^ { 2 } } \right] \geq 1 - \frac { 2 } { k } .
$$

Proof. Let $C ^ { \prime } > 0$ b<sub>e a su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e cons</sub>t<sub>an</sub>t<sub>.</sub> F<sub>or a</sub>ll $i \in [ d ]$ , since $\pmb { \sigma } _ { i } ^ { 2 } \sim \mathrm { i n v } \chi ^ { 2 } ( 1 , 1 )$ <sub>we</sub> h<sub>ave</sub> f<sub>or</sub> $t \geq C ^ { \prime }$ th<sub>a</sub>t $\mathrm { P r } [ \sigma _ { i } ^ { 2 } \geq t ] = C ( t ) t ^ { - 1 / 2 }$ <sub>, w</sub>ith $C ( t ) \ \in \ [ 1 , \sqrt { e } ]$ <sub>.</sub> C<sub>ons</sub>id<sub>er</sub> th<sub>e even</sub>t<sub>s</sub> $H _ { i } : = \left\{ \sigma _ { i } ^ { 2 } \geq d ^ { 2 } / ( 2 k ) ^ { 2 } \right\}$ <sub>an</sub>d l<sub>e</sub>t $\begin{array} { r } { S = \sum _ { i \in [ d ] } \mathbb { 1 } _ { H _ { i } } } \end{array}$ <sub>.</sub> It h<sub>o</sub>ld<sub>s</sub> th<sub>a</sub>t $\operatorname* { P r } [ H _ { i } ] = 2 C k / d$ <sub>,</sub> <sub>an</sub>d th<sub>us</sub> $\mathbb { E } [ S ] = 2 C k$ Since S is the sum of inde<sub>p</sub>endent Bernoulli random variables, Cheb<sub>y</sub>shev’s ine<sub>q</sub>ualit<sub>y y</sub>ields

$$
\begin{array} { r l r } & { \operatorname* { P r } \bigg [ S \le \frac { 1 } { 2 } \mathbb { E } [ S ] \bigg ] \le \operatorname* { P r } \bigg [ | S - \mathbb { E } [ S ] | \ge \frac { 1 } { 2 } \mathbb { E } [ S ] \bigg ] \le \frac { 4 \operatorname { V a r } [ S ] } { \mathbb { E } [ S ] ^ { 2 } } } & \\ & { \quad \le \frac { 4 d \operatorname { P r } [ H _ { 1 } ] } { \mathbb { E } [ S ] ^ { 2 } } = \frac { 4 } { \mathbb { E } [ S ] } \le \frac { 2 } { k } . } & \end{array}
$$

Th<sub>us,</sub> <sub>a</sub>t l<sub>eas</sub>t � <sub>o</sub>f th<sub>e</sub> $\sigma _ { j } ^ { 2 } ,$ <sub>s</sub> <sub>are</sub> <sub>a</sub>t l<sub>eas</sub>t $\frac { d ^ { 2 } } { 4 k ^ { 2 } }$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - { \textstyle { \frac { 2 } { k } } }$

Lemma A.5 (Concentration of the $\chi _ { k } ^ { 2 }$ distribution [LM00]). Let $\mathbf { \boldsymbol { x } } \sim \boldsymbol { \chi } _ { k } ^ { 2 }$ , then for any $t \geq 0$

$$
\operatorname* { P r } [ x \geq k + 2 { \sqrt { k t } } + 2 t ] \leq \exp ( - t ) \quad a n d \quad \operatorname* { P r } [ x \leq k - 2 { \sqrt { k t } } ] \leq \exp ( - t ) .
$$

## B Folklore Algorithm

I<sub>n</sub> thi<sub>s</sub> <sub>sec</sub>ti<sub>on</sub> <sub>we</sub> <sub>exp</sub>l<sub>a</sub>i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>lkl<sub>ore</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> f<sub>or</sub> G<sub>auss</sub>i<sub>an</sub> <sub>mean</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on,</sub> <sub>w</sub>hi<sub>c</sub>h d<sub>emons</sub>t<sub>ra</sub>t<sub>es</sub> the ti<sub>g</sub>htness of Theorem 2.1. The al<sub>g</sub>orithm <sub>p</sub>roceeds in two <sub>p</sub>hases: first, it estimates tr(Σ) usin<sub>g</sub> the avera<sub>g</sub>e norm of half of the sam<sub>p</sub>les, and second, it runs friendl<sub>y</sub>-core with cli<sub>pp</sub>in<sub>g</sub> radius tr(Σ). B<sub>y</sub> th<sub>e guaran</sub>t<sub>ees o</sub>f F<sub>r</sub>i<sub>en</sub>dl<sub>y-core,</sub> th<sub>e resu</sub>lti<sub>ng ou</sub>t<sub>pu</sub>t i<sub>s</sub> th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>mean on</sub> h<sub>a</sub>lf <sub>o</sub>f th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> <sub>p</sub>lus Gaussian noise with variance a<sub>pp</sub>roximatel<sub>y</sub> $\operatorname { t r } ( \Sigma ) / n ^ { 2 }$

Theorem B.1 (Folklore algorithm). For all $\varepsilon , \delta , \beta > 0$ there exists an $( \varepsilon , \delta ) \ – D P$ mechanism M such thatfor all $\mu \in \mathbb { R } ^ { d }$ and positive semi-definite $\Sigma \in \mathbb { R } ^ { d \times d }$ if

$$
n = \Omega \Bigg ( \frac { \log ( 1 / \beta \delta ) } { \varepsilon } \Bigg )
$$

and $X = x _ { 1 } , \ldots , x _ { n } \sim N ( \mu , \Sigma )$ then

$$
\| M ( \pmb { X } ) - \mu \| _ { 2 } \leq O \left( \sqrt { \frac { \operatorname { t r } ( \Sigma ) + \| \Sigma \| _ { o p } \log ( 1 / \beta ) } { n } } + \frac { \sqrt { d \cdot \operatorname { t r } ( \Sigma ) \log ( 1 / \delta ) } \log ( n / \beta ) } { \varepsilon n } \right)
$$

with probability at least $1 - \beta .$

Proof. In order to define the mechanism M we will require the following two results.

We use a sli<sub>g</sub>htl<sub>y</sub> modified version of stable-histo<sub>g</sub>ram from [KV18]. The ori<sub>g</sub>inal version of the statement in Theorem B.2 is from [BGSUZ21], we instantiate it here with bins $\left\{ [ 2 ^ { b - 1 } , 2 ^ { b } ] \right\} _ { b \in \mathbb { Z } }$ an<sup>d</sup> out<sub>p</sub>ut $\tilde { Y } = 2 ^ { \tilde { b } } \in \left\{ 2 ^ { b - 1 } , 2 ^ { b } , 2 ^ { b + 1 } \right\}$ i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f $\tilde { b } \in \{ b - 1 , b , b + 1 \}$

Lemma B.2 (Stable-histogram [BGSUZ21]). There exists an algorithm StableHistogram that is (�, �)-diferentially private and satisfies the following guarantee: Let $z _ { 1 } , \ldots , z _ { n }$ be drawn i.i.d. from distribution �. Suppose that there exists $b \in \mathbb { Z }$ and a constant $\begin{array} { r } { \beta ^ { \prime } < \frac { 1 } { 4 } , } \end{array}$ such that $\operatorname* { P r } [ z _ { i } \notin$ $\left[ 2 ^ { b - 1 } , 2 ^ { b + 1 } \right] ] \leq \beta ^ { \prime }$ for any fixed $i \in [ n ]$ . There exists a constant $C > 0$ such that, for all $0 < \varepsilon , \beta , \delta < 1$ if

$$
n \geq \frac { C } { \varepsilon } \log \frac { 1 } { \beta \delta } ,
$$

then StableHistogram $\ L _ { \varepsilon , \delta } ( z _ { 1 } , \ L _ { \cdot } \dots , z _ { n } ) \in [ 2 ^ { b - 1 } , 2 ^ { b + 1 } ]$ with probability at least $1 - \beta .$

We will also use the friendl<sub>y</sub>-core a<sub>gg</sub>re<sub>g</sub>ation al<sub>g</sub>orithm discovered b<sub>y</sub> [TCKMS22] to estimate the mean. The exact statement of Theorem B.3 does not a<sub>pp</sub>ear ex<sub>p</sub>licitl<sub>y</sub> in [TCKMS22], but it f<sub>o</sub>ll<sub>ows</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> f<sub>rom</sub> th<sub>e</sub>i<sub>r resu</sub>lt<sub>s.</sub>

Lemma B.3 (Friendly-core averaging). Fix �, $\delta , R > 0$ . There exists an $( \varepsilon , \delta ) \ – D P$ algorithm A that gets as input $X = x _ { 1 } , \ldots , x _ { n }$ and satisfies the following: $I f n \geq \tilde { \Omega } \Big ( \sqrt { \log ( 1 / \beta \delta ) } / \varepsilon \Big )$ and $\left\| x _ { i } - x _ { j } \right\| _ { 2 } \leq R$ for all $i , j \in [ n ]$ then

$$
\left\| \mathcal { R } ( X ) - \frac { 1 } { n } \sum x _ { i } \right\| _ { 2 } \leq O \left( \frac { R \sqrt { d \log ( 1 / \delta ) \log ( 1 / \beta ) } } { \varepsilon n } \right)
$$

with probability at least $1 - \beta .$   
ProofofTheorem B.3. Let A denote the algorithm FC\_Avg from [TCKMS22] run with zCDP   
p<sup>arameters</sup> $\rho = \varepsilon ^ { 2 } / 1 0 \log ( 1 / \delta )$ <sub>an</sub>d $\delta _ { 0 } = \delta / 2$ <sub>an</sub>d di<sub>ame</sub>t<sub>er</sub> $R .$   
First we ar<sub>g</sub>ue that A is <sub>p</sub>rivate. $\mathrm { B y }$ Theorem 5.2 of [TCKMS22], al<sub>g</sub>orithm $\mathcal { A }$ i<sub>s</sub> $( \rho , \delta ) \mathrm { \ - z C D P }$   
<sub>w</sub>hi<sub>c</sub>h<sub>,</sub> b<sub>y</sub> F<sub>ac</sub>t 2<sub>.</sub>7 <sub>o</sub>f $\left[ \mathrm { T C K M S 2 2 } \right]$ <sub>w</sub>ith $\delta ^ { \prime }  \delta _ { 0 }$ <sub>,</sub> im<sub>p</sub>lies that $\mathcal { A }$ i<sub>s</sub> $( \rho + 2 \sqrt { \rho \ln ( 1 / \delta _ { 0 } ) } , 2 \delta _ { 0 } ) \ – \mathrm { D P }$   
B<sub>y</sub> our settin<sub>g</sub> of $\rho$ <sub>an</sub>d $\delta _ { 0 }$ <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> $\mathcal { A }$ i<sub>s</sub> $( \varepsilon , \delta ) \mathrm { - D P }$   
To see that A is accurate, we can a<sub>pp</sub>l<sub>y</sub> Claim C.4 of [TCKMS22], which states that if   
$n = \Omega ( \sqrt { \ln ( 1 / \rho \beta \delta _ { 0 } ) / \rho } )$ <sub>an</sub>d $\left\| x _ { i } - x _ { j } \right\| _ { 2 } \leq \mathrm { ~ }$ f<sub>or a</sub>ll $i , j \in [ n ]$ th<sub>en</sub>   
$\left\| \mathcal { I } ( X ) - \frac { 1 } { n } \sum x _ { i } \right\| _ { 2 } \leq O \left( \frac { R } { n } \sqrt { \frac { d \log ( 1 / \beta ) } { \rho } } \right)$   
<sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \beta .$ Usin<sub>g</sub> our settin<sub>g</sub> of $\rho$ <sub>an</sub>d $\delta _ { 0 }$ <sub>conc</sub>l<sub>u</sub>d<sub>es</sub> th<sub>e proo</sub>f<sub>.</sub>   
Note: The requirement on � in Claim C.4 of[TCKMS22] is missing a square root—it reads � $\ge \log ( 1 / \beta \delta ) / \rho ;$ however,   
the proof of Claim C.4 only requires $\iota \geq \sqrt { \log ( 1 / \beta \delta ) / \rho }$ in order to invoke the relevant claim (Claim C.3), and thus the   
requirement on � in our proof is suficient. □

Next, we define the mechanism M. Let $C > 0$ b<sub>e</sub> <sub>a</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>cons</sub>t<sub>an</sub>t<sub>.</sub>

Algorithm 1 Folklore estimator M   
Input: Privacy parameter $\varepsilon , \delta \in ( 0 , 1 )$ <sub>,</sub> d<sub>a</sub>t<sub>ase</sub>t $X = x _ { 1 } , \ldots , x _ { n } ,$ f<sub>a</sub>il<sub>ure</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $\beta > 0 .$   
Output: Estimate $\tilde { \mu } \in \mathbb { R } ^ { d }$   
S<sub>e</sub>t $\begin{array} { r } { \ell \gets \frac { C } { \varepsilon } \log ( 1 / \beta \delta ) } \end{array}$   
P<sub>a</sub>rtiti<sub>o</sub>n � int<sub>o</sub> $X _ { 0 }$ <sub>an</sub>d $X _ { 1 } ,$ <sub>,</sub> <sub>an</sub>d $X _ { 0 }$ into $B _ { 1 } , \ldots , B _ { \ell }$   
F<sub>or eac</sub>h $i \in [ \ell ]$ <sub>an</sub>d $x _ { 1 } , \ldots , x _ { n / 2 \ell } \in B _ { i }$ l<sub>e</sub>t $\begin{array} { r } { y _ { j } = \frac { 1 } { \sqrt { 2 } } \big ( x _ { 2 j - 1 } - x _ { 2 j } \big ) } \end{array}$ f<sub>or eac</sub>h $j \in [ n / 4 \ell ]$   
F<sub>or eac</sub>h $i \in [ \ell ]$ l<sub>e</sub>t   
${ \bar { y } } _ { i } \gets { \frac { 4 \ell } { n } } \sum _ { u : \in B _ { i } }$ �� <sup>2</sup><sub>2</sub>   
$\overline { { y _ { j } \in B } } _ { i }$   
Let $\tilde { y }$ b<sub>e</sub> th<sub>e ou</sub>t<sub>pu</sub>t <sub>o</sub>f St<sub>a</sub>bl<sub>e</sub>Hi<sub>s</sub>t<sub>ogram</sub> $\varepsilon / 2 , \delta / 2$ on <sup>i</sup>n<sub>p</sub>ut $\left( \bar { y } _ { i } \right) _ { i \in [ \ell ] }$   
Let $R  C \sqrt { \tilde { y } \log ( n / \beta ) }$   
<sup>O</sup>ut<sub>p</sub>ut $\tilde { \mu } \gets \mathcal { A } _ { \varepsilon / 2 , \delta / 2 } ( X _ { 1 } , R )$ <sub>w</sub>h<sub>ere</sub> $\mathcal { A }$ i<sub>s</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m</sub> f<sub>rom</sub> Th<sub>eorem</sub> B<sub>.</sub>3   
First, we ar<sub>g</sub>ue that M is <sub>p</sub>rivate. Since stable-histo<sub>g</sub>ram and A are $( \varepsilon / 2 , \delta / 2 )$ <sub>-</sub>DP<sub>,</sub> b<sub>as</sub>i<sub>c</sub>   
com<sub>p</sub>osition and <sub>p</sub>ost<sub>p</sub>rocessin<sub>g</sub> <sub>y</sub>ields that M is $( \varepsilon , \delta ) \mathrm { - D P }$   
N<sub>ex</sub>t<sub>, we s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e accuracy guaran</sub>t<sub>ee</sub> h<sub>o</sub>ld<sub>s.</sub> Th<sub>e</sub> b<sub>as</sub>i<sub>c</sub> id<sub>ea</sub> i<sub>s</sub> t<sub>o s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>eac</sub>h $\bar { y } _ { i }$   
<sub>g</sub>i<sub>v</sub>es a m<sub>u</sub>lti<sub>p</sub>licati<sub>v</sub>e a<sub>pp</sub>roximation to $\operatorname { t r } ( \Sigma )$ <sub>w</sub>ith hi<sub>g</sub>h <sub>pro</sub>b<sub>a</sub>bilit<sub>y,</sub> <sub>an</sub>d th<sub>us,</sub> <sub>s</sub>t<sub>a</sub>bl<sub>e-</sub>hi<sub>s</sub>t<sub>ogram</sub>   
returns an estimate $\tilde { y }$ that is a m<sub>u</sub>lti<sub>p</sub>licati<sub>v</sub>e a<sub>pp</sub>roximation to $\operatorname { t r } ( \Sigma )$ <sub>.</sub> W<sub>e</sub> th<sub>en argue</sub> th<sub>a</sub>t <sub>w</sub>ith   
hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub> $\left\| x _ { i } - x _ { j } \right\| _ { 2 } \leq R$ f<sub>or a</sub>ll $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ <sub>an</sub>d $x _ { j }$ in $X _ { 1 \cdot } \mathrm { A p p l y i n g }$ th<sub>e guaran</sub>t<sub>ee o</sub>f Th<sub>eorem</sub> B<sub>.</sub>3   
<sub>an</sub>d <sub>a un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>su</sub>fi<sub>ces</sub> t<sub>o comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e argumen</sub>t<sub>.</sub> Th<sub>e rema</sub>i<sub>n</sub>d<sub>er o</sub>f th<sub>e proo</sub>f f<sub>orma</sub>li<sub>zes</sub> thi<sub>s</sub>   
int<sub>u</sub>iti<sub>ve a</sub>r<sub>gu</sub>m<sub>e</sub>nt<sub>.</sub>

Lemma B.4 (Concentration of the norm). Let $Y = y _ { 1 } , \dots , y _ { m } \sim N ( 0 , \Sigma )$ then for all $\beta > 0$

$$
\left| \frac { 1 } { m } \sum \| y _ { i } \| _ { 2 } ^ { 2 } - \operatorname { t r } ( \Sigma ) \right| \leq \operatorname { t r } ( \Sigma ) \sqrt { \frac { \log ( 2 / \beta ) } { m } } + \| \Sigma \| _ { o p } \frac { \log ( 2 / \beta ) } { m }
$$

with probability at least $1 - \beta .$

Proof. This follows from rotation invariance of $N ( 0 , \Sigma )$ <sub>an</sub>d Ch<sub>e</sub>b<sub>ys</sub>h<sub>ev</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y.</sub> Ob<sub>serve</sub> th<sub>a</sub>t if $\boldsymbol { y } \sim { \mathcal { N } } ( 0 , \Sigma )$ , then we can e<sub>q</sub>uivalentl<sub>y</sub> write y as $\Sigma ^ { 1 / 2 } z$ <sub>w</sub>h<sub>ere</sub> $z \sim { \mathcal { N } } ( 0 , \mathbb { I } )$ <sub>.</sub> N<sub>ow,</sub> if $\Sigma =$ $V D V ^ { \top }$ th<sub>en</sub> $\| \pmb { y } \| _ { 2 } ^ { 2 } = \Im ^ { \top } V D V ^ { \top } z$ <sub>.</sub> B<sub>y</sub> r<sub>o</sub>t<sub>a</sub>ti<sub>o</sub>n in<sub>va</sub>ri<sub>a</sub>n<sub>ce o</sub>f th<sub>e</sub> G<sub>auss</sub>i<sub>a</sub>n di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>,</sub> th<sub>e</sub> r<sub>a</sub>nd<sub>o</sub>m <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> $\tilde { z } = V ^ { \top } ;$ z is still distributed accordin<sub>g</sub> to $N ( 0 , \mathbb { I } )$ <sub>.</sub> Th<sub>us,</sub> $\begin{array} { r } { \| \pmb { y } \| _ { 2 } ^ { 2 } = \tilde { z } ^ { \top } D \tilde { z } = \sum _ { j \in [ d ] } \lambda _ { j } z [ j ] ^ { 2 } } \end{array}$ <sub>w</sub>h<sub>ere</sub> $\lambda _ { j }$ i<sub>s</sub> th<sub>e</sub> $j ^ { \mathrm { t h } }$ <sub>e</sub>i<sub>genva</sub>l<sub>ue</sub> <sub>o</sub>f $\Sigma$ <sub>an</sub>d $z [ j ] \sim \chi _ { 1 } ^ { 2 }$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the above ar<sub>g</sub>ument to each $\mathbf { \nabla } _ { \mathbf { \boldsymbol { y } } _ { i } }$ <sub>, we</sub> h<sub>ave</sub> th<sub>a</sub>t

$$
\frac { 1 } { m } \sum _ { i \in [ m ] } \| \pmb { y } _ { i } \| _ { 2 } ^ { 2 } = \frac { 1 } { m } \sum _ { i \in [ m ] \atop j \in [ d ] } \lambda _ { j } z _ { i } [ j ] ^ { 2 }
$$

Sin<sub>ce</sub> $\mathbb { E } [ \| y _ { i } \| _ { 2 } ^ { 2 } ] = \operatorname { t r } ( \Sigma )$ <sub>,</sub> <sub>an</sub>d <sub>s</sub>i<sub>nce</sub> th<sub>e</sub> $z _ { i } [ j ] ^ { 2 } s$ <sub>are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t f<sub>or</sub> <sub>eac</sub>h � <sub>an</sub>d $j ,$ we can a<sub>pp</sub>l<sub>y</sub> Scalar Bernstein $( \mathrm { e . g . }$ , Corollar<sub>y</sub> 2.9.2 [Ver26]) to see that

$$
\operatorname* { P r } \left[ \left| \frac { 1 } { m } \sum _ { \stackrel { i \in [ m ] } { j \in [ d ] } } \lambda _ { j } z _ { i } [ j ] ^ { 2 } - \operatorname { t r } ( \Sigma ) \right| \geq \operatorname { t r } ( \Sigma ) \sqrt { \frac { \log ( 2 / \beta ) } { m } } + | | \Sigma | | _ { \mathrm { o p } } \frac { \log ( 2 / \beta ) } { m } \right] \leq \beta . \quad \Sigma
$$

B<sub>y</sub> our settin<sub>g</sub> of <sub>�</sub> and $\ell ,$ <sub>an</sub>d b<sub>y</sub> Th<sub>eorem</sub> B<sub>.</sub>4 <sub>w</sub>ith $m = n / 4 \ell$ <sub>an</sub>d $\beta \gets 1 / 5$ <sub>, we</sub> h<sub>ave</sub> th<sub>a</sub>t Pr[y¯<sub>�</sub> ∉ $\mathrm { t r } ( \Sigma ) \cdot [ 1 / 2 , 2 ] ] \ \leq \ 1 / 5$ . Thus<sub>,</sub> b<sub>y</sub> Theorem B.2<sub>,</sub> and our settin<sub>g</sub> of $n ,$ <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t $\tilde { \boldsymbol { y } } \in \mathrm { t r } ( \Sigma ) \cdot [ 1 / 2 , 2 ]$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \beta .$

Let $B _ { 0 }$ b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t $\tilde { y }$ i<sub>s no</sub>t i<sub>n</sub> thi<sub>s</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>, an</sub>d l<sub>e</sub>t $B _ { 1 }$ b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t <sub>some</sub> $x _ { i } - \mu$ h<sub>as</sub> $\| { \pmb x } _ { i } - { \pmb \mu } \| _ { 2 } \geq C \sqrt { { \mathrm { t r } } ( \Sigma ) \log ( n / \beta ) }$ f<sub>or some</sub> l<sub>arge cons</sub>t<sub>an</sub>t $C > 0$ <sub>.</sub> B<sub>y</sub> Th<sub>eo</sub>r<sub>e</sub>m B<sub>.</sub>4 <sub>w</sub>ith $m = 1$ <sub>an</sub>d th<sub>e un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>over</sub> th<sub>e �</sub> d<sub>a</sub>t<sub>apo</sub>i<sub>n</sub>t<sub>s, we</sub> h<sub>ave</sub> th<sub>a</sub>t $\Pr [ B _ { 1 } ] \leq \beta .$ <sub>.</sub> N<sub>ow, co</sub>nditi<sub>o</sub>n<sub>e</sub>d <sub>o</sub>n $\overline { { B _ { 0 } \cup B _ { 1 } } }$ <sub>we</sub> h<sub>ave</sub> th<sub>a</sub>t $\left\| x _ { i } - x _ { j } \right\| _ { 2 } \leq R$ f<sub>or a</sub>ll � <sub>an</sub>d $j ,$ <sub>an</sub>d th<sub>us,</sub> b<sub>y</sub> Th<sub>eorem</sub> B<sub>.</sub>3<sub>,</sub>

$$
\left. \mathcal { R } ( X ) - \frac { 2 } { n } \sum _ { x _ { i } \in X _ { 1 } } \mathbf { x } _ { i } \right. _ { 2 } \leq O \left( \frac { R \sqrt { d \log ( 1 / \delta ) \log ( 1 / \beta ) } } { \varepsilon n } \right)
$$

<sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - \beta .$ Let $B _ { 2 }$ b<sub>e</sub> th<sub>e</sub> <sub>even</sub>t th<sub>a</sub>t $\tilde { \mu }$ fails to satisf<sub>y</sub> the abo<sub>v</sub>e ine<sub>qu</sub>alit<sub>y</sub>. Then

$$
\operatorname* { P r } [ B _ { 2 } ] = \operatorname* { P r } [ B _ { 2 } \mid B _ { 1 } \cup B _ { 0 } ] \operatorname* { P r } [ B _ { 1 } \cup B _ { 0 } ] + \operatorname* { P r } [ B _ { 2 } \mid \overline { { B _ { 1 } \cup B _ { 0 } } } ] \operatorname* { P r } [ \overline { { B _ { 1 } \cup B _ { 0 } } } ] \leq 3 \beta .
$$

T<sub>o</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e</sub> <sub>argumen</sub>t<sub>,</sub> <sub>we</sub> <sub>nee</sub>d t<sub>o</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>mean</sub> <sub>on</sub> $X _ { 1 }$ i<sub>s</sub> <sub>c</sub>l<sub>ose</sub> t<sub>o</sub> $\mu$ <sub>w</sub>ith hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub>. Let $B _ { 3 }$ b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t

$$
\left\| \frac { 2 } { n } \sum _ { \pmb { x } _ { i } \in X _ { 1 } } \pmb { x } _ { i } - \mu \right\| _ { 2 } \geq C \sqrt { \frac { \mathrm { t r } ( \Sigma ) + \| \Sigma \| _ { \mathrm { o p } } \log ( 1 / \beta ) } { n } } .
$$

Standard ar<sub>gu</sub>ments for the min-max rate of Ga<sub>u</sub>ssian mean estimation im<sub>p</sub>l<sub>y</sub> that Pr $\left[ B _ { 3 } \right] \leq \beta$ Sin<sub>ce</sub> $\tilde { \mu }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> <sub>accuracy</sub> <sub>guaran</sub>t<sub>ee</sub> <sub>o</sub>f Th<sub>eorem</sub> B<sub>.</sub>1 <sub>w</sub>h<sub>enever</sub> <sub>ne</sub>ith<sub>er</sub> <sub>o</sub>f $B _ { 2 }$ or $B _ { 3 }$ <sub>occur,</sub> th<sub>e</sub> <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>over</sub> $B _ { 2 }$ <sub>an</sub>d $B _ { 3 }$ i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t th<sub>e guaran</sub>t<sub>ee</sub> h<sub>o</sub>ld<sub>s w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $1 - \tilde { \beta }$ <sub>w</sub>ith $\tilde { \beta } = 4 \beta$ □