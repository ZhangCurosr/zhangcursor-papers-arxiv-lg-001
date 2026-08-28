# Sharp Minimax Regret for Infinite-Memory Logistic Prediction

Vaneet A<sub>gg</sub>ar<sub>w</sub>al P<sub>ur</sub>d<sub>ue</sub> U<sub>n</sub>i<sub>vers</sub>it<sub>y</sub>

## Abstract

W<sub>e s</sub>t<sub>u</sub>d<sub>y on</sub>li<sub>ne pre</sub>di<sub>c</sub>ti<sub>on</sub> f<sub>or a spec</sub>ifi<sub>c</sub> fi<sub>n</sub>it<sub>e-a</sub>l<sub>p</sub>h<sub>a</sub>b<sub>e</sub>t<sub>, exogenous</sub>l<sub>y</sub> d<sub>r</sub>i<sub>ven source w</sub>ith i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub> i<sub>npu</sub>t <sub>memory.</sub> I<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t R<sub>a</sub>d<sub>emac</sub>h<sub>er</sub> i<sub>npu</sub>t<sub>s</sub> $( U _ { t } )$ <sub>are o</sub>b<sub>serve</sub>d <sub>sequen</sub>ti<sub>a</sub>ll<sub>y, an</sub>d th<sub>e nex</sub>t bi<sub>nary</sub> <sub>mar</sub>k h<sub>as</sub> l<sub>og</sub>it $\textstyle \sum _ { j = 1 } ^ { t } \theta _ { j } U _ { t + 1 - j }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $| \theta _ { j } | \le r _ { j }$ <sub>an</sub>d $\begin{array} { r } { \sum _ { j } r _ { j } \le B . } \end{array}$ <sub>.</sub> R<sub>egre</sub>t i<sub>s expec</sub>t<sub>e</sub>d <sub>cumu</sub>l<sub>a</sub>ti<sub>ve excess</sub> log loss. Lag � can afect prediction by scale $r _ { j }$ <sub>an</sub>d <sub>en</sub>t<sub>ers</sub> <sub>on</sub>l<sub>y</sub> $n _ { T , j } = T - j + 1$ <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>roun</sub>d<sub>s,</sub> l<sub>ea</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> l<sub>ag-reso</sub>l<sub>ve</sub>d <sub>spec</sub>t<sub>rum</sub> $\begin{array} { r } { \Gamma _ { T } ( r ) = \sum _ { j = 1 } ^ { T } \log \Bigl ( 1 + n _ { T , j } r _ { j } ^ { 2 } \Bigr ) } \end{array}$ <sub>.</sub> F<sub>or every summa</sub>bl<sub>e enve</sub>l<sub>ope, a</sub> l<sub>oca</sub>li<sub>ze</sub>d B<sub>ayes</sub>i<sub>an</sub> <sub>m</sub>i<sub>x</sub>t<sub>ure</sub> <sub>proves</sub> $\mathcal { R } _ { T } ( r ) \leq C \Gamma _ { T } ( r )$ <sub>.</sub> F<sub>or exponen</sub>ti<sub>a</sub>l <sub>an</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enve</sub>l<sub>opes, un</sub>d<sub>er</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e</sub> di<sub>mens</sub>i<sub>on</sub> <sub>con</sub>diti<sub>on,</sub> <sub>a</sub> T<sub>oep</sub>lit<sub>z-</sub>d<sub>es</sub>i<sub>gn</sub> <sub>converse</sub> <sub>proves</sub> $\mathcal { R } _ { T } ( r ) \geq c \Gamma _ { T } ( r )$ <sub>,</sub> <sub>w</sub>ith <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> <sub>a</sub>ll<sub>owe</sub>d t<sub>o</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e</sub> fi<sub>xe</sub>d d<sub>ecay</sub> <sub>parame</sub>t<sub>ers</sub> <sub>an</sub>d th<sub>e</sub> l<sub>og</sub>it b<sub>oun</sub>d<sub>.</sub> Th<sub>us</sub> $\Gamma _ { T } ( r )$ i<sub>s</sub> th<sub>e</sub> <sub>m</sub>i<sub>n</sub>i<sub>max</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve-regre</sub>t <sub>sca</sub>l<sub>e</sub> f<sub>or</sub> thi<sub>s</sub> <sub>source</sub> <sub>c</sub>l<sub>ass</sub> i<sub>n</sub> th<sub>ese</sub> <sub>canon</sub>i<sub>ca</sub>l <sub>reg</sub>i<sub>mes,</sub> <sub>g</sub>i<sub>v</sub>i<sub>ng</sub> $\Theta ( \alpha ^ { - 1 } \log ^ { 2 } T )$ f<sub>or</sub> $r _ { j } = A e ^ { - \alpha j }$ <sub>an</sub>d $\Theta ( T ^ { 1 / \bar { ( 2 s ) } } )$ f<sub>or</sub> $r _ { j } = A j ^ { - s } , s > 1$ <sub>.</sub> Th<sub>e converse</sub> i<sub>s spec</sub>ifi<sub>c</sub> t<sub>o</sub> th<sub>e exogenous</sub> l<sub>agge</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d i<sub>s no</sub>t <sub>a pro</sub>fil<sub>e-on</sub>l<sub>y</sub> th<sub>eorem</sub> f<sub>or ar</sub>bit<sub>rary s</sub>t<sub>a</sub>ti<sub>onary</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>e-memory sources.</sub> R<sub>e</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng on</sub>l<sub>y</sub> th<sub>e</sub> <sub>mos</sub>t <sub>recen</sub>t ℎ i<sub>npu</sub>t<sub>s cos</sub>t<sub>s or</sub>d<sub>er</sub> $\begin{array} { r } { \sum _ { j > h } n _ { T , j } \theta _ { j } ^ { 2 } , } \end{array}$ <sub>ye</sub>t th<sub>e same wors</sub>t<sub>-case</sub> t<sub>runca</sub>ti<sub>on pro</sub>fil<sub>e can correspon</sub>d t<sub>o</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>ll<sub>y</sub> dif<sub>eren</sub>t <sub>regre</sub>t<sub>.</sub> A <sub>sca</sub>l<sub>e</sub>d <sub>on</sub>li<sub>ne</sub> N<sub>ew</sub>t<sub>on</sub> <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> <sub>a</sub>tt<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> <sub>spec</sub>t<sub>rum</sub> <sub>upper</sub> b<sub>oun</sub>d<sub>.</sub>

## 1 Introduction

M<sub>any</sub> <sub>sequence</sub> <sub>pre</sub>di<sub>c</sub>t<sub>ors</sub> <sub>use</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t <sub>even</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a-genera</sub>ti<sub>ng</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> h<sub>as</sub> <sub>no</sub> fi<sub>n</sub>it<sub>e</sub> M<sub>ar</sub>k<sub>ov</sub> <sub>or</sub>d<sub>er.</sub> Thi<sub>s occurs</sub> i<sub>n even</sub>t <sub>s</sub>t<sub>reams, ca</sub>t<sub>egor</sub>i<sub>ca</sub>l ti<sub>me ser</sub>i<sub>es w</sub>ith <sub>covar</sub>i<sub>a</sub>t<sub>es, an</sub>d <sub>recurren</sub>t filt<sub>ers: recen</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>ons are re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>exp</sub>li<sub>c</sub>itl<sub>y, w</sub>hil<sub>e o</sub>ld<sub>er o</sub>b<sub>serva</sub>ti<sub>ons are</sub> di<sub>scar</sub>d<sub>e</sub>d <sub>or compresse</sub>d<sub>.</sub> Th<sub>e s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>ques</sub>ti<sub>on</sub> i<sub>s no</sub>t <sub>on</sub>l<sub>y</sub> h<sub>ow accura</sub>t<sub>e</sub>l<sub>y a</sub> fi<sub>n</sub>it<sub>e con</sub>t<sub>ex</sub>t <sub>approx</sub>i<sub>ma</sub>t<sub>es</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>con</sub>diti<sub>ona</sub>l l<sub>aw,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> h<sub>ow</sub> difi<sub>cu</sub>lt it i<sub>s</sub> t<sub>o</sub> l<sub>earn</sub> th<sub>e un</sub>k<sub>nown</sub> l<sub>aw ac</sub>ti<sub>ng on</sub> th<sub>a</sub>t <sub>con</sub>t<sub>ex</sub>t<sub>.</sub> Th<sub>ese</sub> t<sub>wo cos</sub>t<sub>s nee</sub>d <sub>no</sub>t h<sub>ave</sub> th<sub>e same sca</sub>l<sub>e.</sub>

W<sub>e s</sub>t<sub>u</sub>d<sub>y</sub> thi<sub>s ques</sub>ti<sub>on un</sub>d<sub>er</sub> l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> l<sub>oss.</sub> At <sub>roun</sub>d $t ,$ <sub>a causa</sub>l <sub>pre</sub>di<sub>c</sub>t<sub>or o</sub>b<sub>serves</sub> $X _ { 1 : t }$ <sub>an</sub>d <sub>ass</sub>i<sub>gns</sub> <sub>a con</sub>diti<sub>ona</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $q _ { t } ( \cdot \mid X _ { 1 : t } )$ to $X _ { t + 1 }$ <sub>.</sub> If th<sub>e</sub> t<sub>rue source</sub> l<sub>aw</sub> i<sub>s</sub> $P _ { \cdot }$ <sub>, w</sub>ith <sub>con</sub>diti<sub>ona</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $p _ { t } ^ { P } ( \cdot \mid X _ { 1 : t } )$ <sub>,</sub> it<sub>s expec</sub>t<sub>e</sub>d <sub>cumu</sub>l<sub>a</sub>ti<sub>ve regre</sub>t i<sub>s</sub>

$$
{ \tt R e g } _ { T } ( Q ; P ) = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { P } { \bf K L } \Big ( p _ { t } ^ { P } ( \cdot \mid X _ { 1 : t } ) \Big \| q _ { t } ( \cdot \mid X _ { 1 : t } ) \Big ) .\tag{1}
$$

Th<sub>us regre</sub>t i<sub>s</sub> th<sub>e excess</sub> l<sub>og</sub> l<sub>oss re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o a pre</sub>di<sub>c</sub>t<sub>or</sub> th<sub>a</sub>t k<sub>nows</sub> th<sub>e source parame</sub>t<sub>er, no</sub>t <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> b<sub>es</sub>t constant or finite-window rule fitted in hindsi<sub>g</sub>ht. B<sub>y</sub> the chain rule for relative entro<sub>py</sub>, (1) equals KL $( P \| Q )$ for the joint law induced by the sequential predictor. Minimizing its worst-case value over a source class is therefore the avera<sub>g</sub>e minimax redundanc<sub>y</sub> of that class; see Clarke and Barron (1990), Merhav and Feder (1998), and Rissanen (1996). We state this identit<sub>y</sub> formall<sub>y</sub> in Pro<sub>p</sub>osition 3.1.

M<sub>ar</sub>k<sub>ov or</sub>d<sub>er a</sub>l<sub>one canno</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> thi<sub>s regre</sub>t<sub>.</sub> A<sub>n unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d bi<sub>nary or</sub>d<sub>er-</sub>ℎ M<sub>ar</sub>k<sub>ov mo</sub>d<sub>e</sub>l h<sub>as a</sub> <sub>separa</sub>t<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> <sub>parame</sub>t<sub>er</sub> f<sub>or</sub> <sub>every</sub> <sub>con</sub>t<sub>ex</sub>t<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>a</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> <sub>mo</sub>d<sub>e</sub>l <sub>may</sub> <sub>use</sub> <sub>on</sub>l<sub>y</sub> ℎ <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> <sub>s</sub>h<sub>are</sub>d <sub>across a</sub>ll <sub>con</sub>t<sub>ex</sub>t<sub>s.</sub> B<sub>o</sub>th <sub>con</sub>diti<sub>on on a</sub> l<sub>eng</sub>th<sub>-</sub>ℎ bi<sub>nary</sub> hi<sub>s</sub>t<sub>ory an</sub>d th<sub>ere</sub>f<sub>ore encoun</sub>t<sub>er</sub> $2 ^ { h }$ <sub>poss</sub>ibl<sub>e con</sub>t<sub>ex</sub>t<sub>s,</sub> b<sub>u</sub>t th<sub>e</sub>i<sub>r e</sub>f<sub>ec</sub>ti<sub>ve</sub> di<sub>mens</sub>i<sub>ons</sub> dif<sub>er exponen</sub>ti<sub>a</sub>ll<sub>y.</sub> Thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on</sub> i<sub>s a</sub>l<sub>rea</sub>d<sub>y v</sub>i<sub>s</sub>ibl<sub>e</sub> i<sub>n c</sub>l<sub>ass</sub>i<sub>ca</sub>l M<sub>ar</sub>k<sub>ov</sub> redundanc<sub>y</sub> and context-tree <sub>p</sub>rediction (Willems, Shtarkov, and Tjalkens 1995; Atteson 1999), and becomes more consequential for unbounded-memor<sub>y</sub> classes (Wu, Hosseini, and Santhanam 2018). Predictive-state <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>ve ra</sub>t<sub>e–</sub>di<sub>s</sub>t<sub>or</sub>ti<sub>on</sub> th<sub>eor</sub>i<sub>es</sub> lik<sub>ew</sub>i<sub>se emp</sub>h<sub>as</sub>i<sub>ze</sub> th<sub>a</sub>t <sub>use</sub>f<sub>u</sub>l <sub>memory</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y con</sub>diti<sub>ona</sub>

future laws rather than b<sub>y</sub> literal la<sub>g</sub> len<sub>g</sub>th (Crutchfield and Youn<sub>g</sub> 1989; Shalizi and Crutchfield 2001;   
Marzen and Crutchfield 2016). Our aim is to quantif<sub>y</sub> the additional cost of learnin<sub>g</sub> those laws online.

## 1.1 Model and minimax question

W<sub>e</sub> <sub>use</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e-a</sub>l<sub>p</sub>h<sub>a</sub>b<sub>e</sub>t <sub>source</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>an</sub>d l<sub>earn</sub>i<sub>ng</sub> <sub>cos</sub>t<sub>s</sub> <sub>can</sub> b<sub>o</sub>th b<sub>e</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>s</sub>h<sub>a</sub>r<sub>p</sub>l<sub>y.</sub> L<sub>e</sub>t $U _ { 1 } , U _ { 2 } , . . .$ <sub>.</sub> b<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t R<sub>a</sub>d<sub>emac</sub>h<sub>er</sub> i<sub>npu</sub>t<sub>s</sub> <sub>an</sub>d l<sub>e</sub>t th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>sym</sub>b<sub>o</sub>l b<sub>e</sub> $X _ { t } = ( U _ { t } , Y _ { t } ) \in$ $\{ - 1 , + 1 \} \times \{ 0 , 1 \}$ <sub>.</sub> F<sub>or an un</sub>k<sub>nown coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>sequence</sub> $\theta = ( \theta _ { j } ) _ { j \geq 1 }$ <sub>,</sub> th<sub>e nex</sub>t <sub>mar</sub>k <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\mathbb { P } _ { \theta } \left( Y _ { t + 1 } = 1 ~ | ~ X _ { 1 : t } \right) = \sigma \left( \sum _ { j = 1 } ^ { t } \theta _ { j } U _ { t + 1 - j } \right) , \qquad \sigma ( z ) = \frac { 1 } { 1 + e ^ { - z } } .\tag{2}
$$

Th<sub>e</sub> f<sub>res</sub>h i<sub>npu</sub>t $U _ { t + 1 }$ i<sub>s un</sub>if<sub>orm an</sub>d k<sub>nown</sub> i<sub>n</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on, so pre</sub>di<sub>c</sub>ti<sub>ng</sub> $X _ { t + 1 }$ <sub>re</sub>d<sub>uces</sub> t<sub>o pre</sub>di<sub>c</sub>ti<sub>ng</sub> $Y _ { t + 1 }$ <sub>.</sub> If i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub>l<sub>y many</sub> $\theta _ { j }$ <sub>are nonzero,</sub> th<sub>e source</sub> h<sub>as no</sub> fi<sub>n</sub>it<sub>e</sub> i<sub>npu</sub>t<sub>-memory or</sub>d<sub>er.</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d i<sub>n</sub>fi<sub>n</sub>it<sub>e-or</sub>d<sub>er ca</sub>t<sub>egor</sub>i<sub>ca</sub>l models are studied from stationarit<sub>y</sub> and er<sub>g</sub>odicit<sub>y p</sub>ers<sub>p</sub>ectives b<sub>y</sub> Fokianos and Truquet (2019); here the <sub>exogenous</sub> i<sub>npu</sub>t<sub>s ma</sub>k<sub>e</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on geome</sub>t<sub>ry exp</sub>li<sub>c</sub>it<sub>.</sub>

L<sub>e</sub>t $r = ( r _ { j } ) _ { j \geq 1 }$ b<sub>e nonnega</sub>ti<sub>ve an</sub>d <sub>non</sub>i<sub>ncreas</sub>i<sub>ng, w</sub>ith $\begin{array} { r } { \sum _ { j } r _ { j } \le B } \end{array}$ <sub>, an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
\Theta ( r ) = \{ \theta : | \theta _ { j } | \leq r _ { j } { \mathrm { ~ f o r ~ a l l ~ } } j \} .
$$

Th<sub>e</sub> <sub>summa</sub>bilit<sub>y</sub> <sub>assump</sub>ti<sub>on</sub> k<sub>eeps</sub> <sub>every</sub> l<sub>og</sub>it i<sub>n</sub> $[ - B , B ]$ <sub>.</sub> W<sub>r</sub>it<sub>e</sub> $P _ { \theta , T }$ f<sub>or</sub> th<sub>e</sub> l<sub>aw o</sub>f $X _ { 1 : T + 1 }$ under (2). The <sub>quan</sub>tit<sub>y</sub> <sub>s</sub>t<sub>u</sub>di<sub>e</sub>d i<sub>n</sub> thi<sub>s</sub> <sub>paper</sub> i<sub>s</sub>

$$
{ \mathcal { R } } _ { T } ( r ) = \operatorname* { i n f } _ { Q } \operatorname* { s u p } _ { \theta \in \Theta ( r ) } { \mathrm { R e g } } _ { T } ( Q ; P _ { \theta , T } ) ,
$$

th<sub>e m</sub>i<sub>n</sub>i<sub>max cumu</sub>l<sub>a</sub>ti<sub>ve regre</sub>t <sub>over</sub> � <sub>pre</sub>di<sub>c</sub>ti<sub>ons.</sub> F<sub>or</sub> $1 \leq j \leq T$ , lag � appears on $n _ { T , j } = T - j + 1$ <sub>roun</sub>d<sub>s.</sub> Th<sub>e</sub> <sub>cen</sub>t<sub>ra</sub>l <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> i<sub>s</sub> th<sub>e</sub> <sub>re</sub>d<sub>un</sub>d<sub>ancy</sub> <sub>spec</sub>t<sub>rum</sub>

$$
\Gamma _ { T } ( r ) = \sum _ { j = 1 } ^ { T } \log \left( 1 + n _ { T , j } r _ { j } ^ { 2 } \right) .\tag{3}
$$

E<sub>very</sub> l<sub>ag</sub> h<sub>as</sub> t<sub>wo sources o</sub>f <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l difi<sub>cu</sub>lt<sub>y:</sub> h<sub>ow muc</sub>h it <sub>can a</sub>f<sub>ec</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on, measure</sub>d b<sub>y</sub> $r _ { j } .$ <sub>, an</sub>d h<sub>ow</sub> <sub>many roun</sub>d<sub>s rema</sub>i<sub>n</sub> i<sub>n w</sub>hi<sub>c</sub>h it <sub>can</sub> b<sub>e</sub> l<sub>earne</sub>d<sub>, measure</sub>d b<sub>y</sub> $n _ { T , j }$ <sub>.</sub> B<sub>o</sub>th d<sub>ecrease w</sub>ith $j ,$ <sub>an</sub>d th<sub>e</sub>i<sub>r pro</sub>d<sub>uc</sub>t $n _ { T , j } r _ { j } ^ { 2 }$ i<sub>s</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>e</sub>’<sub>s</sub> di<sub>mens</sub>i<sub>on</sub>l<sub>ess</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on sca</sub>l<sub>e.</sub> A <sub>wea</sub>k <sub>coor</sub>di<sub>na</sub>t<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y</sub> thi<sub>s</sub> <sub>sca</sub>l<sub>e, w</sub>hil<sub>e a s</sub>t<sub>rong coor</sub>di<sub>na</sub>t<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> th<sub>e</sub> l<sub>ogar</sub>ith<sub>m o</sub>f it<sub>s a</sub>tt<sub>a</sub>i<sub>na</sub>bl<sub>e reso</sub>l<sub>u</sub>ti<sub>on.</sub>

## 1.2 Main results and technical novelty

Th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> th<sub>eorems</sub> <sub>g</sub>i<sub>ve</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> di<sub>rec</sub>t <sub>answer.</sub> F<sub>or</sub> <sub>every</sub> h<sub>or</sub>i<sub>zon</sub> <sub>an</sub>d <sub>every</sub> <sub>summa</sub>bl<sub>e</sub> <sub>enve</sub>l<sub>ope,</sub> Th<sub>eorem</sub> 4<sub>.</sub>1 <sub>proves</sub>

$$
\boxed { \mathcal { R } _ { T } ( r ) \leq C \Gamma _ { T } ( r ) . }\tag{4}
$$

F<sub>or</sub> th<sub>e</sub> <sub>exponen</sub>ti<sub>a</sub>l <sub>an</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enve</sub>l<sub>opes,</sub> th<sub>e</sub> <sub>converse</sub> i<sub>n</sub> Th<sub>eorem</sub> 4<sub>.</sub>5 <sub>an</sub>d C<sub>oro</sub>ll<sub>ar</sub>i<sub>es</sub> 4<sub>.</sub>8 <sub>an</sub>d 4<sub>.</sub>9 <sub>g</sub>i<sub>ves,</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e</sub> di<sub>mens</sub>i<sub>on</sub> <sub>con</sub>diti<sub>on,</sub>

$$
\boxed { \mathcal { R } _ { T } ( r ) \geq c \Gamma _ { T } ( r ) . }\tag{5}
$$

H<sub>ere</sub> <sub>an</sub>d b<sub>e</sub>l<sub>ow,</sub> <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> i<sub>n</sub> th<sub>e</sub> <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>may</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e</sub> fi<sub>xe</sub>d d<sub>ecay</sub> <sub>parame</sub>t<sub>ers</sub> <sub>an</sub>d <sub>on</sub> �<sub>.</sub> H<sub>ence,</sub> i<sub>n</sub> th<sub>ese reg</sub>i<sub>mes,</sub> $\Gamma _ { T } ( r )$ i<sub>s</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>max cumu</sub>l<sub>a</sub>ti<sub>ve-regre</sub>t <sub>sca</sub>l<sub>e.</sub> Th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>uses</sub> th<sub>e exogenous</sub>

R<sub>a</sub>d<sub>emac</sub>h<sub>er</sub> i<sub>npu</sub>t<sub>s</sub> <sub>an</sub>d th<sub>e</sub>i<sub>r</sub> <sub>over</sub>l<sub>app</sub>i<sub>ng</sub> T<sub>oep</sub>lit<sub>z</sub> d<sub>es</sub>i<sub>gn.</sub> W<sub>e</sub> d<sub>o</sub> <sub>no</sub>t <sub>c</sub>l<sub>a</sub>i<sub>m</sub> th<sub>a</sub>t th<sub>e</sub> <sub>same</sub> <sub>converse</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> <sub>a</sub> <sub>memory-</sub>d<sub>ecay</sub> <sub>pro</sub>fil<sub>e</sub> <sub>a</sub>l<sub>one,</sub> <sub>or</sub> th<sub>a</sub>t it h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> <sub>every</sub> <sub>s</sub>t<sub>a</sub>ti<sub>onary</sub> <sub>source</sub> <sub>w</sub>ith i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub> <sub>memory.</sub>

Th<sub>e</sub> <sub>upper</sub> b<sub>oun</sub>d dif<sub>ers</sub> f<sub>rom</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>ng</sub> <sub>a</sub> <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> <sub>o</sub>f l<sub>eng</sub>th ℎ <sub>an</sub>d <sub>pay</sub>i<sub>ng</sub> ℎ l<sub>og</sub> �<sub>:</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>w</sub>ith $n _ { T , j } r _ { j } ^ { 2 } \ \leq \ 1$ <sub>are</sub> l<sub>e</sub>ft <sub>unreso</sub>l<sub>ve</sub>d <sub>an</sub>d <sub>c</sub>h<sub>arge</sub>d <sub>on</sub>l<sub>y a</sub>t th<sub>e</sub>i<sub>r</sub> t<sub>o</sub>t<sub>a</sub>l <sub>s</sub>i<sub>gna</sub>l <sub>energy.</sub> Th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>e</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>e-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>ana</sub>l<sub>ogue o</sub>f <sub>an</sub>i<sub>so</sub>t<sub>rop</sub>i<sub>c parame</sub>t<sub>r</sub>i<sub>c co</sub>di<sub>ng, an</sub>d it <sub>comp</sub>l<sub>emen</sub>t<sub>s</sub> fi<sub>n</sub>it<sub>e-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>on</sub>li<sub>ne</sub> lo<sub>g</sub>istic-re<sub>g</sub>ression bounds (Hazan, A<sub>g</sub>arwal, and Kale 2007; Foster et al. 2018; Shamir 2020; Jacquet, Shamir, and Sz<sub>p</sub>ankowski 2021). The mixture is an information-theoretic codin<sub>g</sub> construction, not an eficienc<sub>y</sub> claim. O<sub>n</sub> th<sub>e</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>s</sub>id<sub>e,</sub> Th<sub>eorem</sub> 5<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> <sub>an</sub> <sub>exp</sub>li<sub>c</sub>it <sub>on</sub>li<sub>ne</sub> N<sub>ew</sub>t<sub>on</sub> <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub>

$$
\operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathrm { R e g } _ { T } ( \mathrm { O N S } _ { r } ; P _ { \theta , T } ) \le C _ { B } \Gamma _ { T } ( r ) .
$$

Th<sub>e converse</sub> i<sub>s</sub> th<sub>e ma</sub>i<sub>n</sub> t<sub>ec</sub>h<sub>n</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>ep.</sub> Th<sub>eorem</sub> 4<sub>.</sub>3 <sub>conver</sub>t<sub>s</sub> th<sub>e mean-square error o</sub>f <sub>a cons</sub>t<sub>ra</sub>i<sub>ne</sub>d l<sub>o</sub> i<sub>s</sub>ti<sub>c max</sub>i<sub>mum-</sub>lik<sub>e</sub>lih<sub>oo</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub> i<sub>n</sub>t<sub>o a</sub> fi<sub>n</sub>it<sub>e-sam</sub> l<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>w</sub>h<sub>enever</sub> th<sub>e em</sub> i<sub>r</sub>i<sub>ca</sub>l G<sub>ram ma</sub>t<sub>r</sub>i<sub>x</sub> i<sub>s we</sub>ll <sub>con</sub>diti<sub>one</sub>d<sub>.</sub> Th<sub>e</sub> l<sub>agge</sub>d d<sub>es</sub>i<sub>gn</sub> h<sub>ere</sub> i<sub>s an over</sub>l<sub>app</sub>i<sub>ng</sub> T<sub>oep</sub>lit<sub>z ma</sub>t<sub>r</sub>i<sub>x, so</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>-row</sub> <sub>concen</sub>t<sub>ra</sub>ti<sub>on</sub> d<sub>oes</sub> <sub>no</sub>t <sub>app</sub>l<sub>y.</sub> L<sub>emma</sub> 4<sub>.</sub>4 i<sub>ns</sub>t<sub>ea</sub>d <sub>represen</sub>t<sub>s</sub> <sub>every</sub> <sub>o</sub>f<sub>-</sub>di<sub>agona</sub>l G<sub>ram</sub> <sub>sum</sub> b<sub>y</sub> <sub>e</sub>d<sub>ge</sub> <sub>pro</sub>d<sub>uc</sub>t<sub>s</sub> <sub>on a</sub> f<sub>ores</sub>t<sub>;</sub> th<sub>ose pro</sub>d<sub>uc</sub>t<sub>s are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t R<sub>a</sub>d<sub>emac</sub>h<sub>er var</sub>i<sub>a</sub>bl<sub>es.</sub> H<sub>oe</sub>fdi<sub>ng</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y an</sub>d G<sub>ers</sub>h<sub>gor</sub>i<sub>n</sub>’<sub>s</sub> th<sub>eorem</sub> th<sub>en y</sub>i<sub>e</sub>ld <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> i<sub>n</sub> di<sub>mens</sub>i<sub>on</sub> � <sub>w</sub>h<sub>en</sub> � i<sub>s a</sub>t l<sub>eas</sub>t <sub>a cons</sub>t<sub>an</sub>t <sub>mu</sub>lti<sub>p</sub>l<sub>e o</sub>f $J ^ { 2 } \log T$ <sub>.</sub> C<sub>om</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>ese</sub> i<sub>ngre</sub>di<sub>en</sub>t<sub>s,</sub> Th<sub>eorem</sub> 4<sub>.</sub>5 <sub>g</sub>i<sub>ves,</sub> f<sub>or every pos</sub>iti<sub>ve</sub> i<sub>n</sub>iti<sub>a</sub>l bl<sub>oc</sub>k <sub>w</sub>ith $J \le T / 2$ <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> thi<sub>s</sub> di<sub>mens</sub>i<sub>on</sub> <sub>con</sub>diti<sub>on,</sub>

$$
\mathcal { R } _ { T } ( r ) \geq \operatorname* { m a x } \left\{ 0 , \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } \log \bigr ( ( T - J + 1 ) r _ { j } ^ { 2 } \bigr ) - C _ { B } J \right\} ,\tag{6}
$$

<sub>w</sub>h<sub>ere</sub> $C _ { B }$ d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> �<sub>.</sub> Thi<sub>s rou</sub>t<sub>e</sub> i<sub>s nonasymp</sub>t<sub>o</sub>ti<sub>c an</sub>d d<sub>oes no</sub>t <sub>assume</sub> l<sub>oca</sub>l <sub>asymp</sub>t<sub>o</sub>ti<sub>c norma</sub>lit<sub>y or</sub> <sub>a</sub> bl<sub>ac</sub>k<sub>-</sub>b<sub>ox</sub> <sub>spec</sub>t<sub>ra</sub>l th<sub>eorem</sub> f<sub>or</sub> <sub>ran</sub>d<sub>om</sub> T<sub>oep</sub>lit<sub>z</sub> <sub>ma</sub>t<sub>r</sub>i<sub>ces.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>spec</sub>t<sub>rum</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>s</sub>h<sub>arp</sub> <sub>ra</sub>t<sub>es</sub> i<sub>n</sub> C<sub>oro</sub>ll<sub>ar</sub>i<sub>es</sub> 4<sub>.</sub>7 t<sub>o</sub> 4<sub>.</sub>9<sub>:</sub>

$$
\begin{array} { r l } { r _ { j } = 0 ( j > k ) \quad \Longrightarrow \quad \mathcal { R } _ { T } ( r ) = \Theta ( k \log T ) , } \\ { r _ { j } = A e ^ { - \alpha j } \quad \Longrightarrow \quad \mathcal { R } _ { T } ( r ) = \Theta ( \alpha ^ { - 1 } \log ^ { 2 } T ) , } \\ { r _ { j } = A j ^ { - s } , \quad s > 1 \quad \Longrightarrow \quad \mathcal { R } _ { T } ( r ) = \Theta ( T ^ { 1 / ( 2 s ) } ) . } \end{array}
$$

Th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>resu</sub>lt h<sub>as</sub> <sub>no</sub> <sub>ex</sub>t<sub>ra</sub> l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> f<sub>ac</sub>t<sub>or.</sub> A h<sub>ar</sub>d<sub>-</sub>t<sub>runca</sub>ti<sub>on</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>pays</sub> ℎ l<sub>og</sub> � f<sub>or</sub> <sub>a</sub>ll <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>es an</sub>d <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e</sub> l<sub>arger ra</sub>t<sub>e</sub> $T ^ { 1 / ( 2 s ) } ( \log T ) ^ { 1 - 1 / ( 2 s ) }$

## 1.3 Why memory decay does not determine regret

Th<sub>e approx</sub>i<sub>ma</sub>ti<sub>on s</sub>id<sub>e</sub> i<sub>s c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub>d i<sub>n</sub> Th<sub>eorem</sub> 3<sub>.</sub>3<sub>.</sub> D<sub>e</sub>l<sub>e</sub>ti<sub>ng</sub> i<sub>npu</sub>t<sub>s o</sub>ld<sub>er</sub> th<sub>an</sub> $h ,$ <sub>or us</sub>i<sub>ng</sub> th<sub>e</sub> B<sub>ayes-op</sub>ti<sub>ma</sub>l <sub>ru</sub>l<sub>e res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e mos</sub>t <sub>recen</sub>t ℎ i<sub>npu</sub>t<sub>s,</sub> i<sub>ncurs cumu</sub>l<sub>a</sub>ti<sub>ve</sub> l<sub>oss o</sub>f <sub>or</sub>d<sub>er</sub> $\textstyle \sum _ { j > h } n _ { T , j } \theta _ { j } ^ { 2 }$ N<sub>ever</sub>th<sub>e</sub>l<sub>ess,</sub>

$$
\Big | \mathrm { t h e ~ s a m e ~ m e m o r y ~ p r o f l e ~ n e e d ~ n o t ~ i m p l y ~ t h e ~ s a m e ~ m i n i m a x ~ r e g r e t . } \Big |\tag{7}
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>e enve</sub>l<sub>ope</sub> $\Theta ( r )$ <sub>an</sub>d th<sub>e ran</sub>k<sub>-one su</sub>b<sub>c</sub>l<sub>ass</sub> i<sub>n w</sub>hi<sub>c</sub>h $\theta = a r$ f<sub>or one un</sub>k<sub>nown amp</sub>lit<sub>u</sub>d<sub>e</sub> $| a | \leq 1$ h<sub>ave</sub> th<sub>e same wors</sub>t<sub>-case</sub> t<sub>runca</sub>ti<sub>on pro</sub>fil<sub>e.</sub> F<sub>or</sub> $r _ { j } = A j ^ { - s }$ <sub>,</sub> h<sub>owever,</sub> Th<sub>eorem</sub> 4<sub>.</sub>10 <sub>g</sub>i<sub>ves regre</sub>t $\Theta ( T ^ { 1 / ( 2 s ) } )$ f<sub>or</sub> th<sub>e enve</sub>l<sub>ope an</sub>d <sub>on</sub>l<sub>y</sub> $\Theta ( \log T )$ f<sub>or</sub> th<sub>e ran</sub>k<sub>-one su</sub>b<sub>c</sub>l<sub>ass.</sub> M<sub>emory</sub> d<sub>ecay measures</sub> th<sub>e cos</sub>t <sub>o</sub>f f<sub>orge</sub>tti<sub>ng</sub> th<sub>e pas</sub>t<sub>; regre</sub>t <sub>a</sub>l<sub>so measures</sub> h<sub>ow many pre</sub>di<sub>c</sub>ti<sub>ve</sub> l<sub>aws rema</sub>i<sub>n</sub> t<sub>o</sub> b<sub>e</sub> l<sub>earne</sub>d<sub>.</sub>

## 1.4 Contributions

Th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons are summar</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows.</sub>

1<sub>.</sub> W<sub>e</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e</sub> l<sub>ag-reso</sub>l<sub>ve</sub>d <sub>spec</sub>t<sub>rum</sub> $\Gamma _ { T } ( r )$ <sub>as an upper</sub> b<sub>oun</sub>d f<sub>or every summa</sub>bl<sub>e enve</sub>l<sub>ope an</sub>d <sub>as</sub> th<sub>e ma</sub>t<sub>c</sub>hi<sub>ng m</sub>i<sub>n</sub>i<sub>max sca</sub>l<sub>e</sub> f<sub>or</sub> th<sub>e canon</sub>i<sub>ca</sub>l <sub>exponen</sub>ti<sub>a</sub>l <sub>an</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enve</sub>l<sub>opes un</sub>d<sub>er</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e</sub> <sub>con</sub>diti<sub>on.</sub> Thi<sub>s</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e</sub> <sub>s</sub>h<sub>arp</sub> <sub>ra</sub>t<sub>es</sub> $\Theta ( \alpha ^ { - 1 } \log ^ { 2 } T )$ <sub>an</sub>d $\Theta ( T ^ { 1 / ( 2 s ) } )$ ).

2<sub>.</sub> W<sub>e</sub> <sub>prove</sub> th<sub>e</sub> <sub>converse</sub> th<sub>roug</sub>h <sub>a</sub> fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>oun</sub>d f<sub>or</sub> <sub>ran</sub>d<sub>om-</sub>d<sub>es</sub>i<sub>gn</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s</sub> <sub>an</sub>d <sub>a new con</sub>diti<sub>on</sub>i<sub>ng argumen</sub>t f<sub>or</sub> th<sub>e over</sub>l<sub>app</sub>i<sub>ng</sub> T<sub>oep</sub>lit<sub>z</sub> l<sub>ag ma</sub>t<sub>r</sub>i<sub>x.</sub> W<sub>e a</sub>l<sub>so s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e same</sub> t<sub>runca</sub>ti<sub>on</sub> <sub>pro</sub>fil<sub>e</sub> <sub>can</sub> <sub>correspon</sub>d t<sub>o</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>ll<sub>y</sub> dif<sub>eren</sub>t <sub>regre</sub>t<sub>.</sub>

3<sub>.</sub> W<sub>e</sub> <sub>g</sub>i<sub>ve</sub> <sub>a</sub> <sub>pro</sub>fil<sub>e-sca</sub>l<sub>e</sub>d <sub>on</sub>li<sub>ne</sub> N<sub>ew</sub>t<sub>on</sub> <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> <sub>a</sub>tt<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> $O _ { B } ( \Gamma _ { T } ( r ) )$ <sub>,</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith <sub>pro</sub>fil<sub>e</sub> <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d <sub>compresse</sub>d filt<sub>er ex</sub>t<sub>ens</sub>i<sub>ons.</sub>

## 2 Related Work

Th<sub>e c</sub>l<sub>oses</sub>t b<sub>o</sub>di<sub>es o</sub>f <sub>wor</sub>k <sub>s</sub>h<sub>are one or</sub> t<sub>wo</sub> i<sub>ngre</sub>di<sub>en</sub>t<sub>s w</sub>ith thi<sub>s paper, suc</sub>h <sub>as sequen</sub>ti<sub>a</sub>l l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> l<sub>oss,</sub> logistic prediction, or unbounded memory, but not the joint statistical problem. The distinction is easiest to state through the objective. We study the minimax expected cumulative excess log loss

$$
\mathcal { R } _ { T } ( r ) = \operatorname* { i n f } _ { Q } \operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathrm { R e g } _ { T } ( Q ; P _ { \theta , T } ) = \operatorname* { i n f } _ { Q } \operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathrm { K L } ( P _ { \theta , T } | | Q ) .\tag{8}
$$

Th<sub>us</sub> th<sub>e me</sub>t<sub>r</sub>i<sub>c</sub> i<sub>s average m</sub>i<sub>n</sub>i<sub>max re</sub>d<sub>un</sub>d<sub>ancy o</sub>f th<sub>e en</sub>ti<sub>re sequen</sub>ti<sub>a</sub>l l<sub>aw.</sub> It i<sub>s ne</sub>ith<sub>er wors</sub>t<sub>-</sub>l<sub>a</sub>b<sub>e</sub>l <sub>compara</sub>t<sub>or</sub> regret nor the KL risk of a single prediction after a training trajectory. Table 1 reports the nearest quantitative b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub> It<sub>s</sub> fi<sub>rs</sub>t t<sub>wo rows g</sub>i<sub>ve</sub> th<sub>e conven</sub>ti<sub>ona</sub>l <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> l<sub>og-</sub>l<sub>oss sca</sub>l<sub>es</sub> i<sub>n</sub> fi<sub>xe</sub>d<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l<sub>, regu</sub>l<sub>ar</sub> <sub>reg</sub>i<sub>mes;</sub> th<sub>e</sub>i<sub>r</sub> f<sub>ea</sub>t<sub>ure an</sub>d <sub>source assump</sub>ti<sub>ons</sub> dif<sub>er</sub> f<sub>rom ours as exp</sub>l<sub>a</sub>i<sub>ne</sub>d b<sub>e</sub>l<sub>ow.</sub>

Table 1: Quantitative comparison under cumulative lo<sub>g</sub>arithmic loss. The first two rows are standard fi<sub>xe</sub>d<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub> Th<sub>e ra</sub>t<sub>es</sub> f<sub>or</sub> thi<sub>s paper</sub> h<sub>o</sub>ld <sub>un</sub>d<sub>er</sub> th<sub>e con</sub>diti<sub>ons</sub> i<sub>n</sub> C<sub>oro</sub>ll<sub>ar</sub>i<sub>es</sub> 4<sub>.</sub>7 t<sub>o</sub> 4<sub>.</sub>9 <sub>an</sub>d Th<sub>eorem</sub> 4<sub>.</sub>10<sub>.</sub>
<table><tr><td>Source class</td><td>Parameter structure</td><td>Cumulative minimax log-loss regret</td></tr><tr><td>Regular finite-dimensional logistic (Shamir 2020; Jacquet, Shamir, and Szpankowski 2021)</td><td>Fixed d free weights; nondegenerate design</td><td>Θ(d log T)</td></tr><tr><td>Binary order-k Markov (Atteson 1999)</td><td>Fixed  $k ; 2 ^ { k }$  interior transition parameters</td><td> $\Theta ( 2 ^ { k } \log T )$ </td></tr><tr><td>This paper: exogenous logistic, finite lag</td><td> $r _ { j } = 0 \mathrm { f o r } j > k ; k$  shared lag coefficients</td><td>Θ(k log T)</td></tr><tr><td>This paper: exogenous logistic, exponential envelope</td><td> $r _ { j } = A e ^ { - \alpha j }$ </td><td> $\boxed { \Theta ( \alpha ^ { - 1 } \log ^ { 2 } T ) }$ </td></tr><tr><td>This paper: exogenous logistic, polynomial envelope</td><td> $r _ { j } = A j ^ { - s } , s >$  1; coordinate-wise envelope</td><td> $\boxed { \Theta ( T ^ { 1 / ( 2 s ) } ) }$ </td></tr><tr><td>This paper: same polynomial profile, rank-one</td><td> $\theta = a r , | a | \leq 1$  ; one shared amplitude</td><td> $\boxed { \Theta ( \log T ) }$ </td></tr></table>

Th<sub>e</sub> <sub>cen</sub>t<sub>erp</sub>i<sub>ece</sub> <sub>o</sub>f th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> i<sub>s</sub> th<sub>e</sub> <sub>rep</sub>l<sub>acemen</sub>t

![](images/7755060a369971b07af3108861b632eb84eb429bc90de2d9c2e9204b323a4a2e.jpg)

(9)

Th<sub>e new</sub> i<sub>ssue</sub> i<sub>s no</sub>t th<sub>e</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> li<sub>n</sub>k<sub>.</sub> Fi<sub>n</sub>it<sub>e-</sub>di<sub>mens</sub>i<sub>ona</sub>l l<sub>og</sub>i<sub>s</sub>ti<sub>c regre</sub>t i<sub>s a</sub>l<sub>rea</sub>d<sub>y un</sub>d<sub>ers</sub>t<sub>oo</sub>d <sub>a</sub>t th<sub>e</sub> � l<sub>og</sub> � scale (Shamir 2020; Jacquet, Shamir, and Sz<sub>p</sub>ankowski 2021). At horizon �, la<sub>g</sub> $j \leq T$ i<sub>n</sub> th<sub>e presen</sub>t <sub>c</sub>l<sub>ass</sub> h<sub>as</sub> it<sub>s</sub> <sub>own</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e</sub> <sub>e</sub>f<sub>ec</sub>t $r _ { j }$ <sub>an</sub>d <sub>on</sub>l<sub>y</sub> $n _ { T , j } = T - j + 1$ <sub>oppor</sub>t<sub>un</sub>iti<sub>es</sub> t<sub>o</sub> b<sub>e</sub> l<sub>earne</sub>d<sub>.</sub> O<sub>ur</sub> <sub>upper</sub> b<sub>oun</sub>d l<sub>oca</sub>li<sub>zes a</sub>t <sub>eac</sub>h <sub>resu</sub>lti<sub>ng sca</sub>l<sub>e</sub> $n _ { T , j } r _ { j } ^ { 2 } ,$ , and the converse shows that their sum in (9) is unavoidable for the d<sub>epen</sub>d<sub>en</sub>t T<sub>oep</sub>lit<sub>z</sub> d<sub>es</sub>i<sub>gn</sub> i<sub>n</sub> th<sub>e</sub> <sub>canon</sub>i<sub>ca</sub>l f<sub>a</sub>di<sub>ng</sub> <sub>reg</sub>i<sub>mes.</sub> T<sub>o</sub> <sub>our</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge,</sub> <sub>pr</sub>i<sub>or</sub> <sub>wor</sub>k h<sub>as</sub> <sub>no</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d thi<sub>s</sub> l<sub>ag-reso</sub>l<sub>ve</sub>d <sub>m</sub>i<sub>n</sub>i<sub>max c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on, nor</sub> th<sub>e po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>regre</sub>t <sub>separa</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>c</sub>l<sub>asses</sub> <sub>w</sub>ith th<sub>e same</sub> t<sub>runca</sub>ti<sub>on pro</sub>fil<sub>e.</sub>

Finite-dimensional logistic prediction. Online logistic regression supplies strong finite-dimensional upper and lower bounds. Online Newton methods <sub>g</sub>ive lo<sub>g</sub>arithmic com<sub>p</sub>arator re<sub>g</sub>ret on bounded domains (Hazan, A<sub>g</sub>arwal, and Kale 2007); norm-sensitive and minimax lower bounds are develo<sub>p</sub>ed b<sub>y</sub> Foster et al. (2018) and Shamir (2020); and <sub>p</sub>recise as<sub>y</sub>m<sub>p</sub>totics are known for cate<sub>g</sub>orical or fixed feature desi<sub>g</sub>ns (Jacquet, Shamir, and Sz<sub>p</sub>ankowski 2021; Drmota et al. 2026). These results ex<sub>p</sub>lain the first row of the table, but th<sub>ey</sub> d<sub>o no</sub>t <sub>y</sub>i<sub>e</sub>ld th<sub>e rema</sub>i<sub>n</sub>i<sub>ng rows</sub> b<sub>y su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng an</sub> “<sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> di<sub>mens</sub>i<sub>on</sub>” f<sub>or</sub> �<sub>.</sub> Th<sub>e num</sub>b<sub>er o</sub>f <sub>re</sub>l<sub>evan</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>grows</sub> <sub>w</sub>ith �<sub>,</sub> th<sub>e</sub> <sub>over</sub>l<sub>app</sub>i<sub>ng</sub> T<sub>oep</sub>lit<sub>z</sub> f<sub>ea</sub>t<sub>ures</sub> <sub>are</sub> <sub>genera</sub>t<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>source,</sub> <sub>an</sub>d th<sub>e</sub> <sub>cr</sub>it<sub>er</sub>i<sub>on</sub> i<sub>n</sub> (8) is source-avera<sub>g</sub>ed redundanc<sub>y</sub> rather than <sub>p</sub>athwise or worst-label re<sub>g</sub>ret. We use online Newton ste<sub>p</sub>s f<sub>or</sub> th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>ve</sub> b<sub>oun</sub>d i<sub>n</sub> Th<sub>eorem</sub> 5<sub>.</sub>1<sub>;</sub> th<sub>e s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>wor</sub>k i<sub>s</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>e-</sub>l<sub>oca</sub>li<sub>ze</sub>d <sub>spec</sub>t<sub>rum an</sub>d it<sub>s</sub> <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> <sub>source-spec</sub>ifi<sub>c</sub> <sub>converse.</sub>

Markov and context-tree models. This literature is closest in prediction metric but diferent in parameter <sub>s</sub>t<sub>ruc</sub>t<sub>ure.</sub> A<sub>n unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d bi<sub>nary or</sub>d<sub>er-</sub>� M<sub>ar</sub>k<sub>ov source ass</sub>i<sub>gns a separa</sub>t<sub>e</sub> B<sub>ernou</sub>lli <sub>parame</sub>t<sub>er</sub> t<sub>o eac</sub>h <sub>o</sub>f $2 ^ { k }$ hi<sub>s</sub>t<sub>or</sub>i<sub>es.</sub> O<sub>ur</sub> fi<sub>n</sub>it<sub>e-</sub>l<sub>ag</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c source a</sub>l<sub>so uses a</sub> l<sub>eng</sub>th<sub>-</sub>� bi<sub>nary</sub> hi<sub>s</sub>t<sub>ory,</sub> b<sub>u</sub>t ti<sub>es a</sub>ll <sub>con</sub>t<sub>ex</sub>t <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> th<sub>roug</sub>h <sub>on</sub>l<sub>y</sub> � <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub> H<sub>ence</sub> th<sub>e respec</sub>ti<sub>ve</sub> fi<sub>xe</sub>d<sub>-or</sub>d<sub>er re</sub>d<sub>un</sub>d<sub>ancy sca</sub>l<sub>es are</sub> $\Theta ( 2 ^ { k } \log T )$ and Θ(� log �) under standard interiorit<sub>y</sub> conditions (Willems, Shtarkov, and Tjalkens 1995; Atteson 1999). Th<sub>e exponen</sub>ti<sub>a</sub>l dif<sub>erence</sub> i<sub>s no</sub>t <sub>cause</sub>d b<sub>y memory</sub> l<sub>eng</sub>th<sub>;</sub> it i<sub>s cause</sub>d b<sub>y</sub> h<sub>ow pre</sub>di<sub>c</sub>ti<sub>ve</sub> l<sub>aws are s</sub>h<sub>are</sub>d <sub>across</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>es.</sub>

F<sub>or un</sub>b<sub>oun</sub>d<sub>e</sub>d <sub>memory, con</sub>ti<sub>nu</sub>it<sub>y-ra</sub>t<sub>e mo</sub>d<sub>e</sub>l<sub>s con</sub>t<sub>ro</sub>l h<sub>ow muc</sub>h th<sub>e nex</sub>t<sub>-sym</sub>b<sub>o</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on can</sub> chan e when two histories share a lon sufix (Wu, Hosseini, and Santhanam 2018). Such a rate measures <sub>sens</sub>iti<sub>v</sub>it<sub>y</sub> t<sub>o</sub> th<sub>e remo</sub>t<sub>e pas</sub>t<sub>,</sub> b<sub>u</sub>t it d<sub>oes no</sub>t <sub>spec</sub>if<sub>y</sub> h<sub>ow many un</sub>k<sub>nown</sub> di<sub>rec</sub>ti<sub>ons genera</sub>t<sub>e</sub> th<sub>a</sub>t <sub>sens</sub>iti<sub>v</sub>it<sub>y.</sub> C<sub>on</sub>t<sub>ex</sub>t<sub>-parame</sub>t<sub>er</sub>i<sub>ze</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>may</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> <sub>many</sub> <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> <sub>un</sub>k<sub>nown</sub> t<sub>rans</sub>iti<sub>on</sub> l<sub>aws,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>our</sub> <sub>c</sub>l<sub>ass</sub> h<sub>as</sub> <sub>one</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> <sub>s</sub>h<sub>are</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>per</sub> l<sub>ag.</sub> Thi<sub>s</sub> i<sub>s</sub> <sub>w</sub>h<sub>y</sub> <sub>ne</sub>ith<sub>er</sub> M<sub>ar</sub>k<sub>ov</sub> <sub>or</sub>d<sub>er</sub> <sub>nor</sub> <sub>a</sub> <sub>con</sub>ti<sub>nu</sub>it<sub>y</sub> <sub>ra</sub>t<sub>e</sub> <sub>a</sub>l<sub>one</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> the s<sub>p</sub>ectrum in (9).

Prediction from compressed memory. Causal states and predictive rate–distortion characterize exact or loss<sub>y</sub> re<sub>p</sub>resentations of the <sub>p</sub>ast throu<sub>g</sub>h their conditional future laws (Crutchfield and Youn<sub>g</sub> 1989; Shalizi and Crutchfield 2001; Marzen and Crutchfield 2016). Recent com<sub>p</sub>ression-to-<sub>p</sub>rediction results for M<sub>ar</sub>k<sub>ov,</sub> hidd<sub>en</sub> M<sub>ar</sub>k<sub>ov,</sub> <sub>an</sub>d <sub>renewa</sub>l <sub>sources</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> th<sub>e</sub> KL <sub>r</sub>i<sub>s</sub>k <sub>o</sub>f <sub>a</sub> fi<sub>na</sub>l <sub>nex</sub>t<sub>-sym</sub>b<sub>o</sub>l <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>an</sub>d <sub>com</sub>bi<sub>ne</sub> normalized redundanc<sub>y</sub> with a conditional-mutual-information memor<sub>y</sub> term (Han, Jana, and Wu 2023; Han, Jian<sub>g</sub>, and Wu 2024). Those results quantif<sub>y</sub> what information about the <sub>p</sub>ast should be retained. Our cumulative objective char<sub>g</sub>es the full redundanc<sub>y</sub> in (8) and also asks how dificult the retained <sub>p</sub>redictive law i<sub>s</sub> t<sub>o</sub> l<sub>earn.</sub> Th<sub>eorem</sub> 4<sub>.</sub>10 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>a</sub> t<sub>runca</sub>ti<sub>on curve a</sub>l<sub>one canno</sub>t <sub>answer</sub> th<sub>a</sub>t <sub>secon</sub>d <sub>ques</sub>ti<sub>on.</sub>

Infinite-order time series and compressed filters. Infinite-order logistic and multinomial models have been studied for existence, stationarit<sub>y</sub>, and er<sub>g</sub>odicit<sub>y</sub> (Fokianos and Truquet 2019), and non<sub>p</sub>arametric $\operatorname { A R } ( \infty )$ models for estimation risk (Goldenshlu<sub>g</sub>er and Zeevi 2001). These objectives difer from cumulative minimax <sub>re</sub>d<sub>un</sub>d<sub>ancy.</sub> O<sub>ur exogenous</sub> fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon cons</sub>t<sub>ruc</sub>ti<sub>on</sub> i<sub>s</sub> d<sub>es</sub>i<sub>gne</sub>d t<sub>o expose</sub> th<sub>e</sub> l<sub>ag-w</sub>i<sub>se</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> g<sup>eometr</sup>y <sup>and</sup> p<sup>ermit a matchin</sup>g <sup>nonas</sup>y<sup>m</sup>p<sup>totic con</sup>v<sup>erse</sup>.

E<sub>xponen</sub>ti<sub>a</sub>l<sub>-sum approx</sub>i<sub>ma</sub>ti<sub>on an</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e-space precon</sub>diti<sub>on</sub>i<sub>ng prov</sub>id<sub>e use</sub>f<sub>u</sub>l <sub>ways</sub> t<sub>o</sub> i<sub>mp</sub>l<sub>emen</sub>t filt<sub>ers</sub> with slowl<sub>y</sub> deca<sub>y</sub>in<sub>g</sub> kernels (Be<sub>y</sub>lkin and Monzon´ 2010; Marsden and Hazan 2026). We use the same <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>pr</sub>i<sub>nc</sub>i<sub>p</sub>l<sub>e</sub> i<sub>n</sub> Th<sub>eorem</sub> 5<sub>.</sub>5<sub>,</sub> b<sub>u</sub>t d<sub>o no</sub>t id<sub>en</sub>tif<sub>y</sub> filt<sub>er approx</sub>i<sub>ma</sub>ti<sub>on w</sub>ith <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>comp</sub>l<sub>ex</sub>it<sub>y: a</sub> <sub>compac</sub>t <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> th<sub>e</sub> <sub>cos</sub>t <sub>o</sub>f <sub>represen</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>pas</sub>t<sub>,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>re</sub>d<sub>un</sub>d<sub>ancy</sub> <sub>spec</sub>t<sub>rum</sub> <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> th<sub>e</sub> <sub>cos</sub>t <sub>o</sub>f l<sub>earn</sub>i<sub>ng</sub> it<sub>s un</sub>k<sub>nown pre</sub>di<sub>c</sub>ti<sub>ve e</sub>f<sub>ec</sub>t<sub>s.</sub>

## 3 Prediction Problem and Long Memory

This section fixes the prediction objective, introduces the canonical exogenously driven logistic source, <sub>an</sub>d d<sub>e</sub>fi<sub>nes</sub> th<sub>e</sub> l<sub>oss cause</sub>d b<sub>y res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ng pre</sub>di<sub>c</sub>ti<sub>on</sub> t<sub>o recen</sub>t i<sub>npu</sub>t<sub>s.</sub> All l<sub>ogar</sub>ith<sub>ms are na</sub>t<sub>ura</sub>l<sub>, an</sub>d $[ a ] _ { + } = \operatorname* { m a x } \{ a , 0 \}$ <sub>.</sub> F<sub>or nonnega</sub>ti<sub>ve quan</sub>titi<sub>es,</sub> $f \lesssim _ { \zeta }$ � <sup>means</sup> $f \leq C _ { \zeta } g , f \gtrsim _ { \zeta } g$ means $g \lesssim _ { \zeta } f$ <sub>, an</sub>d $f \asymp _ { \zeta } g$ <sub>means</sub> b<sub>o</sub>th<sub>.</sub> S<sub>u</sub>b<sub>scr</sub>i<sub>p</sub>t<sub>s on compar</sub>i<sub>son no</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> th<sub>e parame</sub>t<sub>ers on w</sub>hi<sub>c</sub>h th<sub>e cons</sub>t<sub>an</sub>t<sub>s may</sub> d<sub>epen</sub>d<sub>.</sub>

## 3.1 Prediction objective and redundancy

L<sub>e</sub>t � b<sub>e</sub> <sub>a</sub> l<sub>aw</sub> <sub>on</sub> $X _ { 1 } , \ldots , X _ { T + 1 }$ <sub>over</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e</sub> <sub>a</sub>l<sub>p</sub>h<sub>a</sub>b<sub>e</sub>t<sub>,</sub> <sub>an</sub>d <sub>wr</sub>it<sub>e</sub> $\nu _ { 0 }$ f<sub>or</sub> it<sub>s</sub> k<sub>nown</sub> i<sub>n</sub>iti<sub>a</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> It<sub>s</sub> <sub>one-s</sub>t<sub>ep con</sub>diti<sub>ona</sub>l l<sub>aw</sub> i<sub>s</sub>

$$
p _ { t } ^ { P } ( \cdot \mid X _ { 1 : t } ) = P ( X _ { t + 1 } \in \cdot \mid X _ { 1 : t } ) .
$$

A causal predictor � consists of conditionals $q _ { t } ( \cdot \mid X _ { 1 : t } )$ <sub>.</sub> W<sub>e</sub> t<sub>a</sub>k<sub>e</sub> it<sub>s</sub> i<sub>n</sub>iti<sub>a</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> t<sub>o</sub> <sub>equa</sub>l $\nu _ { 0 } ;$ <sub>a</sub>ll <sub>source</sub> classes considered below share this initial law. This convention only removes an irrelevant constant. The joint law induced by � is

$$
Q ( x _ { 1 : T + 1 } ) = \nu _ { 0 } ( x _ { 1 } ) \prod _ { t = 1 } ^ { T } q _ { t } ( x _ { t + 1 } \mid x _ { 1 : t } ) .
$$

The expected cumulative excess log loss of � under $P$ i<sub>s</sub>

$$
\mathrm { R e g } _ { T } ( Q ; P ) = \mathbb { E } _ { P } \sum _ { t = 1 } ^ { T } \left[ - \log q _ { t } ( X _ { t + 1 } \mid X _ { 1 : t } ) + \log p _ { t } ^ { P } ( X _ { t + 1 } \mid X _ { 1 : t } ) \right] .\tag{10}
$$

Proposition 3.1 (Bayes decomposition and redundancy identity). For every � and every causal predictor $Q ,$ with both sides interpreted in the extended sense,

$$
\mathrm { R e g } _ { T } ( Q ; P ) = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { P } \mathrm { K L } \Big ( p _ { t } ^ { P } ( \cdot \mid X _ { 1 : t } ) \Big \| q _ { t } ( \cdot \mid X _ { 1 : t } ) \Big ) = \mathrm { K L } ( P \| Q ) .\tag{11}
$$

Consequently,for a source class $\mathcal { P } _ { T }$ with a common known initial distribution,

$$
\operatorname* { i n f } _ { Q } \operatorname* { s u p } _ { P \in \mathcal { P } _ { T } } \mathrm { R e g } _ { T } ( Q ; P ) = \operatorname* { i n f } _ { Q } \operatorname* { s u p } _ { P \in \mathcal { P } _ { T } } \mathrm { K L } ( P \| Q ) ,\tag{12}
$$

which is the average minimax redundancy of $\mathcal { P } _ { T }$ .

Th<sub>e</sub> <sub>proo</sub>f i<sub>s</sub> th<sub>e</sub> <sub>c</sub>h<sub>a</sub>i<sub>n</sub> <sub>ru</sub>l<sub>e</sub> f<sub>or</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>en</sub>t<sub>ropy</sub> <sub>an</sub>d i<sub>s</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A f<sub>or</sub> <sub>comp</sub>l<sub>e</sub>t<sub>eness.</sub> I<sub>n</sub> <sub>con</sub>t<sub>ras</sub>t<sub>,</sub> the one-ste<sub>p</sub> risk studied b<sub>y</sub> Han, Jian<sub>g</sub>, and Wu (2024) <sub>p</sub>redicts onl<sub>y</sub> $X _ { T + 1 }$ after seeing a length-� trajectory; for that objective, redundancy is divided by the sample size and accompanied by a memory term. For the cumulative objective (10), redundanc<sub>y</sub> is exactl<sub>y</sub> the value of the <sub>g</sub>ame.

## 3.2 Canonical source and lag-wise information geometry

L<sub>e</sub>t $( U _ { t } ) _ { t \geq 1 }$ b<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>w</sub>ith $\mathbb { P } ( U _ { t } = 1 ) = \mathbb { P } ( U _ { t } = - 1 ) = 1 / 2$ <sub>.</sub> S<sub>e</sub>t $Y _ { 1 } \sim \mathrm { B e r } ( 1 / 2 )$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e o</sub>b<sub>serve</sub>d <sub>sym</sub>b<sub>o</sub>l

$$
X _ { t } = ( U _ { t } , Y _ { t } ) \in \mathcal { X } : = \{ - 1 , + 1 \} \times \{ 0 , 1 \} .
$$

F<sub>or</sub> <sub>a</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>sequence</sub> $\theta = ( \theta _ { j } ) _ { j \geq 1 }$ <sub>,</sub> th<sub>e con</sub>diti<sub>ona</sub>l l<sub>aw o</sub>f th<sub>e nex</sub>t <sub>mar</sub>k i<sub>s</sub>

$$
Y _ { t + 1 } \mid X _ { 1 : t } \sim \operatorname { B e r } ( \sigma ( \eta _ { t + 1 } ( \theta ) ) ) , \qquad \eta _ { t + 1 } ( \theta ) = \sum _ { j = 1 } ^ { t } \theta _ { j } U _ { t + 1 - j } .\tag{13}
$$

W<sub>r</sub>it<sub>e</sub> $P _ { \theta , T }$ for the resulting joint law of $X _ { 1 : T + 1 }$ <sub>.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> <sub>pas</sub>t<sub>,</sub> $U _ { t + 1 }$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>un</sub>if<sub>orm</sub> <sub>an</sub>d $Y _ { t + 1 }$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $U _ { t + 1 }$ . Hence the next-s<sub>y</sub>mbol law on X is

$$
P _ { \theta , T } ( X _ { t + 1 } = ( u , y ) \mid X _ { 1 : t } ) = \frac { 1 } { 2 } \sigma ( \eta _ { t + 1 } ( \theta ) ) ^ { y } \big ( 1 - \sigma ( \eta _ { t + 1 } ( \theta ) ) \big ) ^ { 1 - y } .\tag{14}
$$

Th<sub>e</sub> <sub>ou</sub>t<sub>pu</sub>t hi<sub>s</sub>t<sub>ory</sub> $Y _ { 1 : t }$ d<sub>oes</sub> <sub>no</sub>t <sub>en</sub>t<sub>er</sub> th<sub>e</sub> <sub>con</sub>diti<sub>ona</sub>l l<sub>aw;</sub> it i<sub>s</sub> <sub>none</sub>th<sub>e</sub>l<sub>ess</sub> <sub>par</sub>t <sub>o</sub>f th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>sequence</sub> <sub>an</sub>d i<sub>s ava</sub>il<sub>a</sub>bl<sub>e</sub> t<sub>o</sub> th<sub>e</sub> l<sub>earner.</sub>

L<sub>e</sub>t $r = ( r _ { j } ) _ { j \geq 1 }$ b<sub>e</sub> <sub>a</sub> <sub>nonnega</sub>ti<sub>ve,</sub> <sub>non</sub>i<sub>ncreas</sub>i<sub>ng</sub> <sub>sequence</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub>

$$
\sum _ { j \ge 1 } r _ { j } \le B < \infty .\tag{15}
$$

Th<sub>e coor</sub>di<sub>na</sub>t<sub>e-w</sub>i<sub>se enve</sub>l<sub>ope c</sub>l<sub>ass</sub> i<sub>s</sub>

$$
\Theta ( r ) = \{ \theta : | \theta _ { j } | \leq r _ { j } { \mathrm { ~ f o r ~ a l l ~ } } j \} .\tag{16}
$$

Condition (15) im<sub>p</sub>lies $| \eta _ { t + 1 } ( \theta ) | \le B$ <sup>f</sup>or ever<sub>y</sub> � an<sup>d</sup> ever<sub>y</sub> $\theta \in \Theta ( r )$ <sub>.</sub> C<sub>om</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> <sub>regre</sub>t d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>w</sub>ith P<sub>ropos</sub>iti<sub>on</sub> 3<sub>.</sub>1<sub>,</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>max cumu</sub>l<sub>a</sub>ti<sub>ve regre</sub>t <sub>o</sub>f th<sub>e enve</sub>l<sub>ope</sub> i<sub>s</sub>

$$
\mathcal { R } _ { T } ( r ) = \operatorname* { i n f } _ { Q } \operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathrm { R e g } _ { T } ( Q ; P _ { \theta , T } ) = \operatorname* { i n f } _ { Q } \operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathrm { K L } ( P _ { \theta , T } | | Q ) .\tag{17}
$$

Th<sub>us</sub> $\mathrm { R e g } _ { T } ( Q ; P _ { \theta , T } )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e regre</sub>t <sub>o</sub>f <sub>one pre</sub>di<sub>c</sub>t<sub>or aga</sub>i<sub>ns</sub>t <sub>one source, w</sub>h<sub>ereas</sub> $\mathcal { R } _ { T } ( r )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>m</sub>i<sub>n</sub>i<sub>max va</sub>l<sub>ue over</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>enve</sub>l<sub>ope.</sub> C<sub>ons</sub>t<sub>an</sub>t<sub>s</sub> d<sub>eno</sub>t<sub>e</sub>d b<sub>y</sub> $C _ { B } , c _ { B }$ <sub>may</sub> d<sub>epen</sub>d <sub>on</sub> thi<sub>s</sub> fi<sub>xe</sub>d l<sub>og</sub>it b<sub>oun</sub>d � b<sub>u</sub>t <sub>no</sub>t <sub>on</sub> �<sub>,</sub> th<sub>e ac</sub>ti<sub>ve</sub> di<sub>mens</sub>i<sub>on, or</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>pro</sub>fil<sub>e.</sub> Th<sub>e same sym</sub>b<sub>o</sub>l <sub>may</sub> d<sub>eno</sub>t<sub>e</sub> dif<sub>eren</sub>t <sub>suc</sub>h <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> i<sub>n</sub> dif<sub>eren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>s.</sub>

Lag � appears in the logits for rounds $t = j , j + 1 , . . . , T$ <sub>, so</sub> it<sub>s samp</sub>l<sub>e s</sub>i<sub>ze</sub> i<sub>s</sub>

$$
n _ { T , j } = ( T - j + 1 ) _ { + } .\tag{18}
$$

E<sub>very</sub> l<sub>ag</sub> h<sub>as</sub> t<sub>wo s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l li<sub>m</sub>it<sub>a</sub>ti<sub>ons.</sub> It<sub>s coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>can a</sub>f<sub>ec</sub>t th<sub>e</sub> l<sub>og</sub>it b<sub>y a</sub>t <sub>mos</sub>t $r _ { j }$ <sub>, an</sub>d it <sub>can</sub> b<sub>e</sub> l<sub>earne</sub>d f<sub>rom on</sub>l<sub>y</sub> $n _ { T , j }$ <sub>roun</sub>d<sub>s.</sub> Th<sub>us o</sub>ld<sub>er</sub> l<sub>ags genera</sub>ll<sub>y</sub> h<sub>ave</sub> b<sub>o</sub>th <sub>sma</sub>ll<sub>er a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e e</sub>f<sub>ec</sub>t<sub>s an</sub>d <sub>s</sub>h<sub>or</sub>t<sub>er s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l lif<sub>e</sub>ti<sub>mes.</sub> Th<sub>e</sub>i<sub>r com</sub>bi<sub>ne</sub>d di<sub>mens</sub>i<sub>on</sub>l<sub>ess sca</sub>l<sub>e</sub> i<sub>s</sub> $n _ { T , j } r _ { j } ^ { 2 }$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on sca</sub>l<sub>e, re</sub>d<sub>un</sub>d<sub>ancy</sub> <sub>spec</sub>t<sub>rum, an</sub>d <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> di<sub>mens</sub>i<sub>on</sub>

$$
s _ { T , j } = n _ { T , j } r _ { j } ^ { 2 } , \qquad \Gamma _ { T } ( r ) = \sum _ { j = 1 } ^ { T } \log ( 1 + s _ { T , j } ) , \qquad d _ { T } ( r ) = \sum _ { j = 1 } ^ { T } \operatorname* { m i n } \{ 1 , s _ { T , j } \} .\tag{19}
$$

F<sub>or a un</sub>i<sub>versa</sub>l <sub>cons</sub>t<sub>an</sub>t $c > 0$ <sub>,</sub> th<sub>e e</sub>l<sub>emen</sub>t<sub>ary</sub> i<sub>nequa</sub>liti<sub>es</sub>

$$
c d _ { T } ( r ) \leq \Gamma _ { T } ( r ) \leq d _ { T } ( r ) + \sum _ { j : s _ { T , j } > 1 } \log s _ { T , j }\tag{20}
$$

<sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t $d _ { T } ( r )$ <sub>coun</sub>t<sub>s s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>ll<sub>y ac</sub>ti<sub>ve</sub> di<sub>rec</sub>ti<sub>ons w</sub>hil<sub>e</sub> $\Gamma _ { T } ( r )$ <sub>a</sub>l<sub>so recor</sub>d<sub>s</sub> th<sub>e</sub>i<sub>r reso</sub>l<sub>u</sub>ti<sub>ons.</sub>

Quadratic information geometry. Let $\psi ( z ) = \log ( 1 + e ^ { z } )$ <sub>.</sub> Th<sub>e</sub> B<sub>ernou</sub>lli l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> f<sub>am</sub>il<sub>y</sub> i<sub>s a canon</sub>i<sub>ca</sub>l <sub>exponen</sub>ti<sub>a</sub>l f<sub>am</sub>il<sub>y</sub> <sub>w</sub>ith $\psi ^ { \prime } ( z ) = \sigma ( z )$ <sub>an</sub>d $\psi ^ { \prime \prime } ( z ) = \sigma ( z ) ( 1 - \sigma ( z ) )$ <sub>.</sub> S<sub>e</sub>t

$$
\kappa _ { B } = \operatorname* { m i n } _ { | z | \leq B } \psi ^ { \prime \prime } ( z ) = \sigma ( B ) ( 1 - \sigma ( B ) ) > 0 .\tag{21}
$$

Lemma 3.2 (Pairwise KL geometry). For any $\theta , \theta ^ { \prime } \in \Theta ( r )$

$$
\frac { \kappa _ { B } } { 2 } \sum _ { j = 1 } ^ { T } n _ { T , j } ( \theta _ { j } - \theta _ { j } ^ { \prime } ) ^ { 2 } \le \mathrm { K L } ( P _ { \theta , T } \| P _ { \theta ^ { \prime } , T } ) \le \frac { 1 } { 8 } \sum _ { j = 1 } ^ { T } n _ { T , j } ( \theta _ { j } - \theta _ { j } ^ { \prime } ) ^ { 2 } .\tag{22}
$$

The upper bound remains valid without the bounded-logit assumption.

Th<sub>e proo</sub>f<sub>, g</sub>i<sub>ven</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A<sub>, com</sub>bi<sub>nes</sub> th<sub>e qua</sub>d<sub>ra</sub>ti<sub>c curva</sub>t<sub>ure o</sub>f th<sub>e</sub> B<sub>ernou</sub>lli l<sub>og-par</sub>titi<sub>on</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>w</sub>ith

$$
\mathbb { E } \left( \sum _ { j = 1 } ^ { t } a _ { j } U _ { t + 1 - j } \right) ^ { 2 } = \sum _ { j = 1 } ^ { t } a _ { j } ^ { 2 } .
$$

Thi<sub>s exac</sub>t <sub>or</sub>th<sub>ogona</sub>lit<sub>y</sub> i<sub>s</sub> th<sub>e reason</sub> th<sub>e pro</sub>fil<sub>e</sub> i<sub>s an</sub>i<sub>so</sub>t<sub>rop</sub>i<sub>c an</sub>d <sub>a</sub>dditi<sub>ve across</sub> l<sub>ags.</sub> It <sub>w</sub>ill <sub>power</sub> b<sub>o</sub>th th<sub>e</sub> <sub>memory</sub> th<sub>eorem an</sub>d th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d<sub>-m</sub>i<sub>x</sub>t<sub>ure upper</sub> b<sub>oun</sub>d<sub>.</sub>

## 3.3 Predictive memory: what truncation does and does not measure

Thi<sub>s su</sub>b<sub>sec</sub>ti<sub>on</sub> i<sub>so</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e approx</sub>i<sub>ma</sub>ti<sub>on s</sub>id<sub>e o</sub>f th<sub>e pro</sub>bl<sub>em.</sub> W<sub>e s</sub>t<sub>u</sub>d<sub>y</sub> t<sub>wo re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>no</sub>ti<sub>ons.</sub> Th<sub>e</sub> fi<sub>rs</sub>t <sub>compares</sub> th<sub>e</sub> t<sub>rue</sub> <sub>source</sub> <sub>w</sub>ith th<sub>e</sub> <sub>source</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> d<sub>e</sub>l<sub>e</sub>ti<sub>ng</sub> <sub>a</sub>ll <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> <sub>a</sub>ft<sub>er</sub> l<sub>ag</sub> ℎ<sub>.</sub> Th<sub>e</sub> <sub>secon</sub>d i<sub>s</sub> <sub>opera</sub>ti<sub>ona</sub>l<sub>:</sub> it <sub>as</sub>k<sub>s</sub> f<sub>or</sub> th<sub>e</sub> B<sub>ayes-op</sub>ti<sub>ma</sub>l <sub>pre</sub>di<sub>c</sub>t<sub>or among a</sub>ll <sub>ru</sub>l<sub>es</sub> th<sub>a</sub>t <sub>re</sub>t<sub>a</sub>i<sub>n on</sub>l<sub>y</sub> th<sub>e mos</sub>t <sub>recen</sub>t ℎ i<sub>npu</sub>t<sub>s.</sub> B<sub>o</sub>th <sub>are</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> th<sub>e</sub> <sub>same</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d t<sub>a</sub>il <sub>energy.</sub>

F<sub>o</sub>r <sub>a</sub> n<sub>o</sub>nn<sub>ega</sub>ti<sub>ve</sub> int<sub>ege</sub>r $h ,$ l<sub>e</sub>t

$$
\boldsymbol { \theta } ^ { ( h ) } = ( \theta _ { 1 } , \dots , \theta _ { h } , 0 , 0 , \dots )
$$

<sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>runca</sub>ti<sub>on</sub> bi<sub>as</sub>

$$
\begin{array} { r } { \mathcal { B } _ { T } ( \theta , h ) = \mathrm { K L } ( P _ { \theta , T } \| P _ { \theta ^ { ( h ) } , T } ) . } \end{array}\tag{23}
$$

Th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d t<sub>a</sub>il <sub>energy</sub> i<sub>s</sub>

$$
V _ { T } ( \theta , h ) = \sum _ { j > h } n _ { T , j } \theta _ { j } ^ { 2 } .\tag{24}
$$

T<sub>o</sub> d<sub>e</sub>fi<sub>ne</sub> <sub>recen</sub>t<sub>-</sub>i<sub>npu</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on,</sub> l<sub>e</sub>t

$$
\mathcal { G } _ { t , h } = \left\{ \begin{array} { l l } { \sigma \big ( U _ { ( t - h + 1 ) \vee 1 } , \ldots , U _ { t } \big ) , } & { h \geq 1 , } \\ { \{ \emptyset , \Omega \} , } & { h = 0 , } \end{array} \right.
$$

b<sub>e</sub> th<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> <sub>mos</sub>t <sub>recen</sub>t $h$ i<sub>npu</sub>t<sub>s</sub> b<sub>e</sub>f<sub>ore</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>ng</sub> $Y _ { t + 1 }$ <sub>.</sub> Th<sub>e</sub> B<sub>ayes-op</sub>ti<sub>ma</sub>l ℎ<sub>-</sub>i<sub>npu</sub>t <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> i<sub>s</sub>

$$
\begin{array} { r } { p _ { \theta , t } ^ { \star , h } = \mathbb { P } _ { \theta } \left( Y _ { t + 1 } = 1 \mid \mathcal { G } _ { t , h } \right) = \mathbb { E } \left[ \sigma ( \eta _ { t + 1 } ( \theta ) ) \mid \mathcal { G } _ { t , h } \right] . } \end{array}\tag{25}
$$

It<sub>s cumu</sub>l<sub>a</sub>ti<sub>ve</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> l<sub>oss</sub> i<sub>s</sub>

$$
\mathcal { D } _ { T } ( \theta , h ) = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { \theta } \mathrm { K L } \Big ( \mathrm { B e r } ( \sigma ( \eta _ { t + 1 } ( \theta ) ) ) \Big \| \mathrm { B e r } ( p _ { \theta , t } ^ { \star , h } ) \Big ) .\tag{26}
$$

B<sub>y</sub> conditional Ba<sub>y</sub>es o<sub>p</sub>timalit<sub>y</sub>, (26) is the smallest ex<sub>p</sub>ected lo<sub>g</sub>-loss excess amon<sub>g</sub> all <sub>p</sub>redictors measurable <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> $\mathcal { G } _ { t , h }$ <sub>a</sub>t <sub>roun</sub>d �<sub>.</sub>

Theorem 3.3 (Two-sided predictive-memory theorem). For every $\theta \in \Theta ( r )$ , every horizon �, and every nonnegative integer ℎ,

$$
\frac { \kappa _ { B } } { 2 } V _ { T } ( \theta , h ) \leq { \mathcal { B } } _ { T } ( \theta , h ) \leq \frac { 1 } { 8 } V _ { T } ( \theta , h ) ,\tag{27}
$$

$$
2 \kappa _ { B } ^ { 2 } V _ { T } ( \theta , h ) \leq \mathcal { D } _ { T } ( \theta , h ) \leq \frac { 1 } { 8 } V _ { T } ( \theta , h ) .\tag{28}
$$

In particular, truncating the parameter and optimally retaining only the last ℎ inputs have the same cumulative order ofpredictive distortion.

Th<sub>e</sub> <sub>proo</sub>f i<sub>s</sub> <sub>g</sub>i<sub>ven</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub> Th<sub>e</sub> t<sub>runca</sub>ti<sub>on</sub> b<sub>oun</sub>d f<sub>o</sub>ll<sub>ows</sub> di<sub>rec</sub>tl<sub>y</sub> f<sub>rom</sub> L<sub>emma</sub> 3<sub>.</sub>2<sub>.</sub> F<sub>or</sub> <sub>recen</sub>t<sub>-</sub>i<sub>npu</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on,</sub> th<sub>e</sub> l<sub>og</sub>it i<sub>s sp</sub>lit i<sub>n</sub>t<sub>o re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>an</sub>d <sub>om</sub>itt<sub>e</sub>d t<sub>erms.</sub> L<sub>og</sub>i<sub>s</sub>ti<sub>c curva</sub>t<sub>ure g</sub>i<sub>ves</sub> th<sub>e upper</sub> b<sub>oun</sub>d<sub>, w</sub>hil<sub>e</sub> Pinsker’s ine<sub>q</sub>ualit<sub>y</sub> and the lower bound on the lo<sub>g</sub>istic slo<sub>p</sub>e over [−�, �] <sub>g</sub>ive the converse.

Remark 3.4 (Scope of the operational memory statement). The sigma-field $\mathcal { G } _ { t , h }$ contains the last ℎ exogenous inputs, not the last ℎ complete marked observations ${ \cal { X } } _ { s } ~ = ~ ( U _ { s } , Y _ { s } )$ <sub>.</sub> R<sub>ecen</sub>t <sub>mar</sub>k<sub>s can carry</sub> i<sub>n</sub>di<sub>rec</sub>t i<sub>n</sub>f<sub>orma</sub>ti<sub>on a</sub>b<sub>ou</sub>t <sub>o</sub>ld<sub>er</sub> i<sub>npu</sub>t<sub>s, so</sub> Th<sub>eorem</sub> 3<sub>.</sub>3 d<sub>oes no</sub>t <sub>c</sub>l<sub>a</sub>i<sub>m op</sub>ti<sub>ma</sub>lit<sub>y among a</sub>ll <sub>compresse</sub>d <sub>s</sub>t<sub>a</sub>t<sub>es o</sub>f th<sub>e</sub> f<sub>u</sub>ll <sub>mar</sub>k<sub>e</sub>d hi<sub>s</sub>t<sub>ory.</sub> It <sub>g</sub>i<sub>ves</sub> <sub>an</sub> <sub>exac</sub>t th<sub>eorem</sub> f<sub>or</sub> i<sub>npu</sub>t<sub>-w</sub>i<sub>n</sub>d<sub>ow</sub> t<sub>runca</sub>ti<sub>on,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> th<sub>e</sub> <sub>memory</sub> <sub>no</sub>ti<sub>on</sub> naturall<sub>y p</sub>aired with the convolutional <sub>p</sub>redictor in (13).

For comparison with the coordinate envelope, define the rank-one, or ray, class

$$
\Theta _ { \mathrm { r a y } } ( r ) = \{ \theta = a r : | a | \leq 1 \} .\tag{29}
$$

Corollary 3.5 (Worst-case envelope profile). For the coordinate-wise envelope $\Theta ( r )$

$$
\operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathcal { B } _ { T } ( \theta , h ) \asymp _ { B } \operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathcal { D } _ { T } ( \theta , h ) \asymp _ { B } \sum _ { j > h } n _ { T , j } r _ { j } ^ { 2 } .\tag{30}
$$

The same statement holdsfor the rank-one class in (29).

Proof. The upper bounds follow from Theorem 3.3 and $\theta _ { j } ^ { 2 } \le r _ { j } ^ { 2 }$ <sub>.</sub> Th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s are a</sub>tt<sub>a</sub>i<sub>ne</sub>d<sub>, up</sub> t<sub>o</sub> th<sub>e</sub> <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> i<sub>n</sub> th<sub>e</sub> th<sub>eorem,</sub> b<sub>y</sub> $\theta = r$ <sub>, w</sub>hi<sub>c</sub>h b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> b<sub>o</sub>th <sub>c</sub>l<sub>asses.</sub> □

Remark 3.6 (Exact versus approximate memory). If infinitely many $\theta _ { j }$ <sub>are nonzero,</sub> th<sub>e exac</sub>t <sub>con</sub>diti<sub>ona</sub>l l<sub>aw</sub> in (13) de<sub>p</sub>ends on arbitraril<sub>y</sub> old in<sub>p</sub>uts. Nevertheless, Theorem 3.3 shows that finite-window <sub>p</sub>rediction <sub>can</sub> b<sub>e accura</sub>t<sub>e w</sub>h<sub>enever</sub> th<sub>e we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t t<sub>a</sub>il i<sub>s sma</sub>ll<sub>.</sub> F<sub>or</sub> fi<sub>xe</sub>d �<sub>,</sub> th<sub>e enve</sub>l<sub>ope</sub> $r _ { j } = A e ^ { - \alpha j }$ <sub>g</sub><sup>i</sup>ves <sub>a w</sub>i<sub>n</sub>d<sub>ow o</sub>f <sub>or</sub>d<sub>er</sub> $\alpha ^ { - 1 } \log ( T / \varepsilon )$ f<sub>or cumu</sub>l<sub>a</sub>ti<sub>ve</sub> di<sub>s</sub>t<sub>or</sub>ti<sub>on a</sub>t <sub>mos</sub>t <sub>�, w</sub>h<sub>ereas</sub> $r _ { j } = A j ^ { - s } , s > 1$ , <sub>g</sub><sup>i</sup>ves a <sub>w</sub>i<sub>n</sub>d<sub>ow o</sub>f <sub>or</sub>d<sub>er</sub> $( T / \varepsilon ) ^ { 1 / ( 2 s - 1 ) }$ <sub>.</sub> Th<sub>ese are approx</sub>i<sub>ma</sub>ti<sub>on s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>s;</sub> Th<sub>eorem</sub> 4<sub>.</sub>10 <sub>w</sub>ill <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>ey</sub> d<sub>o</sub> <sub>no</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>max regre</sub>t <sub>o</sub>f th<sub>e source c</sub>l<sub>ass.</sub>

## 4 Information-Theoretic Minimax Regret and the Redundancy Spectrum

W<sub>e now</sub> t<sub>urn</sub> f<sub>rom</sub> th<sub>e approx</sub>i<sub>ma</sub>ti<sub>on</sub> l<sub>oss</sub> i<sub>n</sub> Th<sub>eorem</sub> 3<sub>.</sub>3 t<sub>o</sub> th<sub>e cos</sub>t <sub>o</sub>f l<sub>earn</sub>i<sub>ng</sub> th<sub>e un</sub>k<sub>nown coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub> B<sub>y</sub> P<sub>ropos</sub>iti<sub>on</sub> 3<sub>.</sub>1<sub>,</sub> thi<sub>s cos</sub>t <sub>can</sub> b<sub>e ana</sub>l<sub>yze</sub>d <sub>as e</sub>ith<sub>er m</sub>i<sub>n</sub>i<sub>max cumu</sub>l<sub>a</sub>ti<sub>ve regre</sub>t <sub>or average m</sub>i<sub>n</sub>i<sub>max re</sub>d<sub>un</sub>d<sub>ancy.</sub> Th<sub>e conc</sub>l<sub>us</sub>i<sub>ons</sub> h<sub>ave</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>scopes:</sub>

$$
\begin{array} { r } { \mathcal { R } _ { T } ( r ) \leq C \Gamma _ { T } ( r ) \quad \mathrm { f o r e v e r y ~ s u m m a b l e ~ e n v e l o p e } , } \end{array}
$$

<sub>w</sub>h<sub>ereas</sub> th<sub>e</sub> <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> l<sub>ower</sub> b<sub>oun</sub>d h<sub>o</sub>ld<sub>s</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>ac</sub>ti<sub>ve</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> fit i<sub>ns</sub>id<sub>e</sub> th<sub>e</sub> <sub>nonasymp</sub>t<sub>o</sub>ti<sub>c</sub> T<sub>oep</sub>lit<sub>z-</sub>d<sub>es</sub>i<sub>gn</sub> <sub>reg</sub>i<sub>me.</sub> I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> th<sub>e</sub> <sub>exponen</sub>ti<sub>a</sub>l <sub>an</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enve</sub>l<sub>opes</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>

$$
c \Gamma _ { T } ( r ) \leq \mathcal { R } _ { T } ( r ) \leq C \Gamma _ { T } ( r )
$$

<sub>un</sub>d<sub>er</sub> th<sub>e con</sub>diti<sub>ons s</sub>t<sub>a</sub>t<sub>e</sub>d b<sub>e</sub>l<sub>ow, w</sub>ith <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> th<sub>a</sub>t <sub>may</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e</sub> fi<sub>xe</sub>d d<sub>ecay parame</sub>t<sub>ers an</sub>d <sub>on</sub> �<sub>.</sub> Th<sub>us</sub> $\Gamma _ { T } ( r )$ i<sub>s</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>max sca</sub>l<sub>e</sub> f<sub>or</sub> th<sub>e canon</sub>i<sub>ca</sub>l <sub>enve</sub>l<sub>opes</sub> i<sub>n</sub> thi<sub>s exogenous</sub>l<sub>y</sub> d<sub>r</sub>i<sub>ven</sub> l<sub>agge</sub>d <sub>source.</sub> Th<sub>e</sub> <sub>converse</sub> i<sub>s no</sub>t <sub>asser</sub>t<sub>e</sub>d f<sub>or ar</sub>bit<sub>rary s</sub>t<sub>a</sub>ti<sub>onary processes</sub> th<sub>a</sub>t <sub>s</sub>h<sub>are on</sub>l<sub>y a memory-</sub>d<sub>ecay pro</sub>fil<sub>e.</sub>

Th<sub>e proo</sub>f h<sub>as</sub> th<sub>ree par</sub>t<sub>s.</sub> A l<sub>oca</sub>li<sub>ze</sub>d B<sub>ayes</sub>i<sub>an co</sub>d<sub>e g</sub>i<sub>ves</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on-</sub>th<sub>eore</sub>ti<sub>c ac</sub>hi<sub>eva</sub>bilit<sub>y.</sub> A fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>oun</sub>d <sub>an</sub>d <sub>a</sub> T<sub>oep</sub>lit<sub>z concen</sub>t<sub>ra</sub>ti<sub>on</sub> l<sub>emma</sub> t<sub>oge</sub>th<sub>er g</sub>i<sub>ve one source-</sub>l<sub>eve</sub>l <sub>converse, ra</sub>th<sub>er</sub> th<sub>an</sub> t<sub>wo separa</sub>t<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s.</sub> W<sub>e</sub> th<sub>en eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e spec</sub>t<sub>rum an</sub>d <sub>prove</sub> th<sub>e memory-pro</sub>fil<sub>e</sub> <sub>separa</sub>ti<sub>on.</sub> C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y exp</sub>li<sub>c</sub>it <sub>pre</sub>di<sub>c</sub>t<sub>ors are</sub> d<sub>e</sub>f<sub>erre</sub>d t<sub>o</sub> S<sub>ec</sub>ti<sub>on</sub> 5<sub>.</sub>

## 4.1 Information-theoretic achievability via Bayesian coding

L<sub>e</sub>t $\pi = \bigotimes _ { j \ge 1 } \operatorname { U n i f } [ - r _ { j } , r _ { j } ]$ <sub>, w</sub>ith <sub>coor</sub>di<sub>na</sub>t<sub>es sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> $r _ { j } = 0$ fi<sub>xe</sub>d <sub>a</sub>t <sub>zero, an</sub>d l<sub>e</sub>t $\pi _ { T }$ d<sub>eno</sub>t<sub>e</sub> it<sub>s</sub> fi<sub>rs</sub>t<sub>-</sub>� <sub>marg</sub>i<sub>na</sub>l<sub>.</sub> Si<sub>nce a</sub> h<sub>or</sub>i<sub>zon-</sub>� <sub>source</sub> l<sub>aw</sub> d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> th<sub>e</sub> fi<sub>rs</sub>t � <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s,</sub> th<sub>e correspon</sub>di<sub>ng</sub> B<sub>ayes</sub>i<sub>an</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y ass</sub>i<sub>gnmen</sub>t i<sub>s</sub>

$$
Q _ { \pi _ { T } } = \int P _ { \nu , T } \pi _ { T } ( d \nu ) .\tag{31}
$$

It is causal because every mixture of joint laws can be factorized into posterior predictive conditionals. The infinite product prior makes these finite-horizon assignments projectively consistent, so the same Bayesian <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> i<sub>s any</sub>ti<sub>me.</sub>

Theorem 4.1 (Anisotropic Bayesian coding bound). There is a universal constant � such that, for every � and every envelope �,

$$
\operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathrm { K L } ( P _ { \theta , T } \| Q _ { \pi _ { T } } ) \leq \frac { 1 } { 2 } \Gamma _ { T } ( r ) + C d _ { T } ( r ) \leq C \Gamma _ { T } ( r ) .\tag{32}
$$

Consequently,

$$
\begin{array} { r } { \mathcal { R } _ { T } ( r ) \leq C \Gamma _ { T } ( r ) . } \end{array}\tag{33}
$$

The constant does not depend on the logit bound �.

Th<sub>e</sub> <sub>proo</sub>f <sub>uses</sub> th<sub>e</sub> <sub>var</sub>i<sub>a</sub>ti<sub>ona</sub>l i<sub>nequa</sub>lit<sub>y</sub>

$$
\mathrm { K L } \left( P \Big \| \int Q _ { \nu } \pi ( d \nu ) \right) \leq \operatorname* { i n f } _ { \rho \ll \pi } \left\{ \int \mathrm { K L } ( P \| Q _ { \nu } ) \rho ( d \nu ) + \mathrm { K L } ( \rho \| \pi ) \right\} .\tag{34}
$$

F<sub>or a coor</sub>di<sub>na</sub>t<sub>e w</sub>ith $s _ { T , j } \ \leq \ 1$ <sub>,</sub> <sub>we</sub> <sub>se</sub>t th<sub>e</sub> <sub>var</sub>i<sub>a</sub>ti<sub>ona</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>equa</sub>l t<sub>o</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> <sub>an</sub>d <sub>pay</sub> $O ( s _ { T , j } )$ <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>w</sub>ith <sub>zero</sub> l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> <sub>cos</sub>t<sub>.</sub> F<sub>or</sub> $s _ { T , j } > 1$ <sub>, we res</sub>t<sub>r</sub>i<sub>c</sub>t th<sub>e coor</sub>di<sub>na</sub>t<sub>e</sub> t<sub>o an</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>o</sub>f l<sub>eng</sub>th $n _ { T , j } ^ { - 1 / 2 }$ adjacent to the true parameter. This costs $\frac { 1 } { 2 }$ <sup>l</sup>o<sub>g</sub> $s _ { T , j } + O ( 1 )$ nats and leaves �(1) pairwise KL. Summing the coordinate costs and a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Lemma 3.2 <sub>g</sub>ives (32). The com<sub>p</sub>lete calculation a<sub>pp</sub>ears in A<sub>pp</sub>endix B.

Remark 4.2 (Why hard truncation is suboptimal). A truncation analysis with an ℎ-dimensional isotropic <sup>re</sup>g<sup>ret</sup> <sup>term</sup> g<sup>i</sup>v<sup>es</sup>

$$
h \log T + \sum _ { j > h } n _ { T , j } r _ { j } ^ { 2 } .
$$

Th<sub>e</sub> <sub>m</sub>i<sub>x</sub>t<sub>ure</sub> th<sub>eorem</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>c</sub>h<sub>arges</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> $j$ <sup>b</sup><sub>y</sub> $\log ( 1 + n _ { T , j } r _ { j } ^ { 2 } )$ <sub>.</sub> C<sub>oor</sub>di<sub>na</sub>t<sub>es</sub> b<sub>e</sub>l<sub>ow no</sub>i<sub>se</sub> l<sub>eve</sub>l <sub>are no</sub>t l<sub>oca</sub>li<sub>ze</sub>d<sub>;</sub> th<sub>e</sub>i<sub>r</sub> t<sub>o</sub>t<sub>a</sub>l <sub>cos</sub>t i<sub>s</sub> th<sub>e</sub>i<sub>r s</sub>i<sub>gna</sub>l <sub>energy.</sub> Thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on removes</sub> th<sub>e ex</sub>t<sub>ra</sub> l<sub>ogar</sub>ith<sub>m</sub> f<sub>or po</sub>l<sub>ynom</sub>i<sub>a</sub>l enve<sup>l</sup>o<sub>p</sub>es.

## 4.2 Converse: conditioning the lagged design

Th<sub>e source-</sub>l<sub>eve</sub>l l<sub>ower</sub> b<sub>oun</sub>d i<sub>s o</sub>bt<sub>a</sub>i<sub>ne</sub>d i<sub>n</sub> th<sub>ree s</sub>t<sub>eps.</sub> W<sub>e</sub> fi<sub>rs</sub>t <sub>g</sub>i<sub>ve a reusa</sub>bl<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> b<sub>oun</sub>d f<sub>or a</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c exper</sub>i<sub>men</sub>t <sub>w</sub>ith <sub>an exogenous ran</sub>d<sub>om</sub> d<sub>es</sub>i<sub>gn.</sub> W<sub>e</sub> th<sub>en ver</sub>if<sub>y</sub> it<sub>s</sub> G<sub>ram-con</sub>diti<sub>on</sub>i<sub>ng</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> f<sub>or</sub> th<sub>e over</sub>l<sub>app</sub>i<sub>ng</sub> l<sub>ag ma</sub>t<sub>r</sub>i<sub>x.</sub> A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>resu</sub>lt <sub>w</sub>ith th<sub>e secon</sub>d <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> Th<sub>eorem</sub> 4<sub>.</sub>5<sub>.</sub>

Ingredient 1: information from a conditioned logistic design. Let $Z \in \mathbb { R } ^ { N \times J }$ h<sub>ave rows</sub> $z _ { i } ^ { \top }$ <sub>,</sub> l<sub>e</sub>t $b \in \mathbb { R } ^ { N }$ and suppose the joint law of $( Z , b )$ d<sub>oes no</sub>t d<sub>epen</sub>d <sub>on</sub> th<sub>e parame</sub>t<sub>er.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> $( Z , b )$ <sub>, o</sub>b<sub>serve</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t l<sub>a</sub>b<sub>e</sub>l<sub>s</sub>

$$
Y _ { i } \mid Z , b , \vartheta \sim \mathrm { B e r } \big ( \sigma ( b _ { i } + z _ { i } ^ { \top } \vartheta ) \big ) , \qquad i = 1 , \ldots , N .\tag{35}
$$

F<sub>or</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>ra</sub>dii $r _ { 1 } , \ldots , r _ { J }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>rec</sub>t<sub>ang</sub>l<sub>e</sub>

$$
K _ { J } = \prod _ { j = 1 } ^ { J } [ - r _ { j } / 2 , r _ { j } / 2 ] , \qquad \Delta _ { J } ^ { 2 } = \sum _ { j = 1 } ^ { J } r _ { j } ^ { 2 } ,\tag{36}
$$

<sub>wr</sub>it<sub>e</sub> $P _ { \vartheta } ^ { ( N ) }$ for the joint law of $( Z , b , Y _ { 1 : N } )$ <sub>an</sub>d

$$
\Re _ { N } ( K _ { J } ) = \operatorname* { i n f } _ { { \mathcal { Q } } } \operatorname* { s u p } _ { \vartheta \in K _ { J } } \mathrm { K L } ( P _ { \vartheta } ^ { ( N ) } \| { \mathcal { Q } } ) .\tag{37}
$$

Theorem 4.3 (Conditioned-design information bound). Fix $B _ { 0 } < \infty , \tau \geq 1 , \mu \in ( 0 , 1 ]$ , and $\delta \in [ 0 , 1 ]$ Suppose that

$$
\operatorname* { m a x } _ { 1 \leq i \leq N } \operatorname* { s u p } _ { \nu \in K _ { J } } | b _ { i } + z _ { i } ^ { \top } \nu | \leq B _ { 0 } \quad a l m o s t s u r e l y ,\tag{38}
$$

$$
\begin{array} { r } { \mathbb { E } \operatorname { t r } ( Z ^ { \top } Z ) \leq \tau N J , } \end{array}\tag{39}
$$

$$
\begin{array} { r } { \mathbb { P } \big ( \lambda _ { \operatorname* { m i n } } ( Z ^ { \top } Z ) \ge \mu N \big ) \ge 1 - \delta , } \end{array}\tag{40}
$$

and assume the bad-design contribution obeys

$$
\Delta _ { J } ^ { 2 } \delta \leq \frac { \tau J } { N } .\tag{41}
$$

If Θ is uniform on $K _ { J }$ and $\boldsymbol { D } = ( Z , b , Y _ { 1 : N } )$ , then, writing $I ( \Theta ; D )$ for mutual information,

$$
I ( \Theta ; D ) \geq \left[ \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } \log ( N r _ { j } ^ { 2 } ) - C _ { B _ { 0 } , \mu , \tau } J \right] _ { + } .\tag{42}
$$

Consequently, the same lower bound holdsfor $\Re _ { N } ( K _ { J } )$

Th<sub>e re</sub>d<sub>un</sub>d<sub>ancy–capac</sub>it<sub>y an</sub>d <sub>es</sub>ti<sub>ma</sub>ti<sub>on rou</sub>t<sub>e</sub> i<sub>n</sub> thi<sub>s</sub> th<sub>eorem</sub> i<sub>s c</sub>l<sub>ass</sub>i<sub>ca</sub>l<sub>; a c</sub>l<sub>ose</sub>l<sub>y re</sub>l<sub>a</sub>t<sub>e</sub>d fi<sub>n</sub>it<sub>e-</sub> dimensional ar<sub>g</sub>ument is used for online lo<sub>g</sub>istic re<sub>g</sub>ression b<sub>y</sub> Shamir (2020). The useful <sub>p</sub>oint is modularit<sub>y</sub>. O<sub>nce</sub> th<sub>e</sub> l<sub>og</sub>it<sub>s are</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>,</sub> th<sub>e converse re</sub>d<sub>uces</sub> t<sub>o a</sub> t<sub>race</sub> b<sub>oun</sub>d <sub>an</sub>d <sub>a</sub> hi<sub>g</sub>h<sub>-pro</sub>b<sub>a</sub>bilit<sub>y</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l G<sub>ram ma</sub>t<sub>r</sub>i<sub>x.</sub> Th<sub>e proo</sub>f <sub>uses a cons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>max</sub>i<sub>mum-</sub>lik<sub>e</sub>lih<sub>oo</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>or an</sub>d <sub>an</sub> <sub>en</sub>t<sub>ropy–es</sub>ti<sub>ma</sub>ti<sub>on</sub> i<sub>nequa</sub>lit<sub>y, w</sub>ith <sub>no</sub> l<sub>oca</sub>l <sub>asymp</sub>t<sub>o</sub>ti<sub>c norma</sub>lit<sub>y assump</sub>ti<sub>on; see</sub> A<sub>ppen</sub>di<sub>x</sub> D<sub>.</sub>

Ingredient 2: conditioning the overlapping lag matrix. The lower bound uses the last $N = T - J + 1$ <sub>roun</sub>d<sub>s on w</sub>hi<sub>c</sub>h th<sub>e</sub> fi<sub>rs</sub>t � l<sub>ags are a</sub>ll <sub>presen</sub>t<sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> $N \times J$ l<sub>ag</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub>

$$
Z _ { t , j } ^ { ( J ) } = U _ { J + t - j } , \qquad 1 \leq t \leq N , \quad 1 \leq j \leq J .\tag{43}
$$

E<sub>ac</sub>h <sub>row</sub> i<sub>s</sub> <sub>a</sub> <sub>consecu</sub>ti<sub>ve</sub> bl<sub>oc</sub>k <sub>o</sub>f th<sub>e</sub> i<sub>npu</sub>t <sub>s</sub>t<sub>ream.</sub> Th<sub>e</sub> <sub>rows</sub> <sub>are</sub> d<sub>epen</sub>d<sub>en</sub>t<sub>,</sub> b<sub>u</sub>t th<sub>e</sub> <sub>o</sub>f<sub>-</sub>di<sub>agona</sub>l G<sub>ram</sub> <sub>sums</sub> h<sub>ave</sub> <sub>a</sub> <sub>spec</sub>i<sub>a</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure.</sub>

Lemma 4.4 (Toeplitz Gram concentration). For every $J \le T$ and $N = T - J + 1$

$$
\mathbb { P } \Bigg ( \frac { N } { 2 } I _ { J } \preceq ( Z ^ { ( J ) } ) ^ { \top } Z ^ { ( J ) } \preceq \frac { 3 N } { 2 } I _ { J } \Bigg ) \geq 1 - 2 J ^ { 2 } \exp \left( - \frac { N } { 8 J ^ { 2 } } \right) .\tag{44}
$$

Here $I _ { J }$ is the $J \times J$ identity matrix.

F<sub>or</sub> fi<sub>xe</sub>d $j \neq k$ <sub>,</sub> th<sub>e correspon</sub>di<sub>ng</sub> G<sub>ram en</sub>t<sub>ry</sub> i<sub>s a sum o</sub>f <sub>var</sub>i<sub>a</sub>bl<sub>es o</sub>f th<sub>e</sub> f<sub>orm</sub> $U _ { s } U _ { s + d } .$ <sub>w</sub>h<sub>ere</sub> $d = | j - k |$ Th<sub>e</sub> <sub>grap</sub>h <sub>w</sub>ith <sub>ver</sub>ti<sub>ces</sub> i<sub>n</sub>d<sub>exe</sub>d b<sub>y</sub> i<sub>npu</sub>t ti<sub>mes</sub> <sub>an</sub>d <sub>e</sub>d<sub>ges</sub> $( s , s + d )$ i<sub>s</sub> <sub>a</sub> f<sub>ores</sub>t<sub>.</sub> Ch<sub>oos</sub>i<sub>ng</sub> <sub>one</sub> <sub>roo</sub>t <sub>s</sub>i<sub>gn</sub> i<sub>n</sub> each component and all edge products is a bijective reparameterization of the vertex signs; hence the edge products are jointly independent Rademachers. Hoefding controls each of-diagonal entry, and a union bound followed b<sub>y</sub> Gersh<sub>g</sub>orin’s theorem <sub>y</sub>ields (44). A full <sub>p</sub>roof is <sub>g</sub>iven in A<sub>pp</sub>endix C.

## Source-level conclusion.

Theorem 4.5 (Spectrum lower bound for the lagged source). For every fixed $B < \infty ,$ , there is a constant $C _ { B } \geq 1$ with the following property. Let $1 \leq J \leq T / 2$ satisfy $r _ { J } > 0 ,$ , set $N = T - J + 1$ , and suppose

$$
N \geq C _ { B } J ^ { 2 } \log ( 2 N ) .\tag{45}
$$

Then

$$
\mathcal { R } _ { T } ( r ) \geq \left[ \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } \log ( N r _ { j } ^ { 2 } ) - C _ { B } J \right] _ { + } .\tag{46}
$$

Th<sub>eorem</sub> 4<sub>.</sub>5 f<sub>o</sub>ll<sub>ows</sub> b<sub>y</sub> <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> Th<sub>eorem</sub> 4<sub>.</sub>3 t<sub>o</sub> th<sub>e</sub> l<sub>as</sub>t � l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>an</sub>d th<sub>e</sub> fi<sub>rs</sub>t � l<sub>ag</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> F<sub>or</sub> the Toe<sub>p</sub>litz matrix in (43), $\mathrm { t r } ( ( Z ^ { ( J ) } ) ^ { \top } Z ^ { ( J ) } ) = N J$ d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y, w</sub>hil<sub>e</sub> L<sub>emma</sub> 4<sub>.</sub>4 <sub>g</sub>i<sub>ves</sub> $\mu = 1 / 2$ <sub>an</sub>d f<sub>a</sub>il<sub>ure</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $\delta \le 2 J ^ { 2 } \exp ( - N / ( 8 J ^ { 2 } ) )$ <sub>.</sub> O<sub>n</sub> th<sub>e</sub> h<sub>a</sub>lf<sub>-enve</sub>l<sub>ope</sub> <sub>rec</sub>t<sub>ang</sub>l<sub>e,</sub> <sub>every</sub> l<sub>og</sub>it i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> $B / 2$ <sub>an</sub>d $\Delta _ { J } ^ { 2 } \leq B ^ { 2 }$ <sub>.</sub> Enl<sub>a</sub>r<sub>g</sub>in<sub>g</sub> $C _ { B }$ in (45) ensures $B ^ { 2 } \delta \leq J / N$ <sub>, so a</sub>ll h<sub>ypo</sub>th<sub>eses o</sub>f th<sub>e genera</sub>l <sub>converse</sub> h<sub>o</sub>ld<sub>.</sub> Th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> <sub>exper</sub>i<sub>men</sub>t i<sub>s</sub> <sub>a</sub> <sub>measura</sub>bl<sub>e</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> f<sub>u</sub>ll <sub>o</sub>b<sub>serve</sub>d <sub>sequence,</sub> <sub>an</sub>d th<sub>e</sub> <sub>pro</sub>d<sub>uc</sub>t<sub>-un</sub>if<sub>orm</sub> <sub>pr</sub>i<sub>or</sub> i<sub>s suppor</sub>t<sub>e</sub>d i<sub>ns</sub>id<sub>e</sub> $\Theta ( r ) ;$ ; redundanc<sub>y</sub>–ca<sub>p</sub>acit<sub>y</sub> and data <sub>p</sub>rocessin<sub>g</sub> therefore <sub>g</sub>ive (46). The conditionin<sub>g</sub> <sub>an</sub>d i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>s</sub>t<sub>eps</sub> <sub>are</sub> <sub>wr</sub>itt<sub>en</sub> <sub>ou</sub>t i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> D<sub>.</sub>

Corollary 4.6 (Strongly active coordinates). There are constants $a _ { B } > 1 , c _ { B } > 0$ , and $C _ { B } \geq 1$ such that the following holds. Define

$$
J _ { T } ( a _ { B } ) = \operatorname* { m a x } \{ j \leq T / 2 : T r _ { j } ^ { 2 } \geq a _ { B } \} ,\tag{47}
$$

with the maximum equal to zero ifthe set is empty. If

$$
T \geq C _ { B } J _ { T } ( a _ { B } ) ^ { 2 } \log ( 2 T ) ,\tag{48}
$$

then

$$
\mathcal { R } _ { T } ( r ) \geq c _ { B } \sum _ { j = 1 } ^ { J _ { T } ( a _ { B } ) } \log ( 1 + T r _ { j } ^ { 2 } ) .\tag{49}
$$

Proof. If $J _ { T } ( a _ { B } ) = 0 .$ <sub>,</sub> th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> <sub>nonnega</sub>ti<sub>v</sub>it<sub>y</sub> <sub>o</sub>f <sub>regre</sub>t<sub>.</sub> Oth<sub>erw</sub>i<sub>se,</sub> <sub>app</sub>l<sub>y</sub> Th<sub>eorem</sub> 4<sub>.</sub>5 <sub>w</sub>ith $J =$ $J _ { T } ( a _ { B } )$ <sub>an</sub>d <sub>use</sub> $N \ge T / 2$ <sub>.</sub> Ch<sub>oose</sub> $a _ { B }$ l<sub>arge enoug</sub>h th<sub>a</sub>t<sub>,</sub> f<sub>or a</sub>ll $\begin{array} { r } { x \ge a _ { B } , \frac 1 2 \log ( x / 2 ) - C _ { B } \ge c _ { B } \log ( 1 + x ) } \end{array}$ □

Th<sub>e upper an</sub>d l<sub>ower</sub> b<sub>oun</sub>d<sub>s are</sub> d<sub>e</sub>lib<sub>era</sub>t<sub>e</sub>l<sub>y s</sub>t<sub>a</sub>t<sub>e</sub>d i<sub>n a</sub> f<sub>orm</sub> th<sub>a</sub>t <sub>exposes</sub> th<sub>e</sub>i<sub>r</sub> d<sub>oma</sub>i<sub>n o</sub>f <sub>va</sub>lidit<sub>y.</sub> F<sub>or an</sub> <sub>ar</sub>bit<sub>rary</sub> <sub>enve</sub>l<sub>ope,</sub> <sub>a</sub> l<sub>arge</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> <sub>o</sub>f b<sub>are</sub>l<sub>y</sub> <sub>ac</sub>ti<sub>ve</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>may</sub> <sub>no</sub>t b<sub>e</sub> <sub>covere</sub>d b<sub>y</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-samp</sub>l<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>.</sub> F<sub>or</sub> <sub>exponen</sub>ti<sub>a</sub>l <sub>an</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>pro</sub>fil<sub>es</sub> <sub>w</sub>ith <sub>summa</sub>bl<sub>e</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s,</sub> h<sub>owever,</sub> th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> $T r _ { j } ^ { 2 } \gtrsim 1$ h<sub>ave car</sub>di<sub>na</sub>lit<sub>y</sub> $o ( \sqrt { T / \log T } )$ <sub>, an</sub>d th<sub>e s</sub>t<sub>rong</sub>l<sub>y ac</sub>ti<sub>ve par</sub>t <sub>carr</sub>i<sub>es a cons</sub>t<sub>an</sub>t f<sub>rac</sub>ti<sub>on o</sub>f $\Gamma _ { T } ( r )$ <sub>.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>su</sub>b<sub>sec</sub>ti<sub>on</sub> <sub>ver</sub>ifi<sub>es</sub> thi<sub>s</sub> <sub>an</sub>d <sub>o</sub>bt<sub>a</sub>i<sub>ns</sub> <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> <sub>ra</sub>t<sub>es.</sub>

## 4.3 Consequences: sharp rates and memory-profile separation

Th<sub>e</sub> <sub>upper</sub> <sub>an</sub>d l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> <sub>now</sub> <sub>re</sub>d<sub>uce</sub> th<sub>e</sub> <sub>m</sub>i<sub>n</sub>i<sub>max</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> t<sub>o</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> $\Gamma _ { T } ( r )$ <sub>.</sub> W<sub>e</sub> fi<sub>rs</sub>t d<sub>o</sub> <sub>so</sub> f<sub>or</sub> fi<sub>n</sub>it<sub>e-,</sub> <sub>exponen</sub>ti<sub>a</sub>l<sub>-, an</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>-memory enve</sub>l<sub>opes.</sub> W<sub>e</sub> th<sub>en compare</sub> th<sub>ese</sub> l<sub>earn</sub>i<sub>ng cos</sub>t<sub>s w</sub>ith th<sub>e</sub>i<sub>r</sub> t<sub>runca</sub>ti<sub>on</sub> <sub>pro</sub>fil<sub>es</sub> <sub>an</sub>d <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>memory</sub> d<sub>ecay</sub> <sub>a</sub>l<sub>one</sub> <sub>canno</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> <sub>regre</sub>t<sub>.</sub>

Sharp rates for canonical envelopes. The conditions in the following corollary are finite-sample versions <sub>o</sub>f th<sub>e requ</sub>i<sub>remen</sub>t th<sub>a</sub>t th<sub>e s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>ll<sub>y ac</sub>ti<sub>ve coor</sub>di<sub>na</sub>t<sub>es</sub> fit i<sub>ns</sub>id<sub>e</sub> th<sub>e</sub> T<sub>oep</sub>lit<sub>z-</sub>d<sub>es</sub>i<sub>gn reg</sub>i<sub>me o</sub>f Th<sub>eorem</sub> 4<sub>.</sub>5<sub>.</sub> Th<sub>e are au</sub>t<sub>oma</sub>ti<sub>c</sub> f<sub>or</sub> fi<sub>xe</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>arame</sub>t<sub>ers as</sub> $T \to \infty$

Corollary 4.7 (Finite memory). There are constants $a _ { B } > 1 , c _ { B } > 0 ,$ , and $C _ { B } \geq 1$ such that thefollowing holds. Suppose $r _ { j } = 0 f o r j > k . \ I f k \leq T / 2$

$$
T \geq C _ { B } k ^ { 2 } \log ( 2 T ) \quad a n d \quad T r _ { k } ^ { 2 } \geq a _ { B } ,\tag{50}
$$

then

$$
c _ { B } \sum _ { j = 1 } ^ { k } \log ( 1 + T r _ { j } ^ { 2 } ) \leq \mathcal { R } _ { T } ( r ) \leq C \sum _ { j = 1 } ^ { k } \log ( 1 + T r _ { j } ^ { 2 } ) .\tag{51}
$$

In particular, for fixed � and fixed positive radii $r _ { 1 } , \dots , r _ { k } , \mathcal { R } _ { T } ( r ) = \Theta ( k \log T )$

Th<sub>e</sub> fi<sub>xe</sub>d<sub>-</sub>� <sub>conc</sub>l<sub>us</sub>i<sub>on recovers</sub> th<sub>e</sub> f<sub>am</sub>ili<sub>ar parame</sub>t<sub>r</sub>i<sub>c re</sub>d<sub>un</sub>d<sub>ancy sca</sub>l<sub>e,</sub> b<sub>u</sub>t th<sub>e mo</sub>d<sub>e</sub>l i<sub>s no</sub>t <sub>a con</sub>t<sub>ex</sub>t t<sub>a</sub>bl<sub>e:</sub> it <sub>s</sub>h<sub>ares</sub> � <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s across a</sub>ll $2 ^ { k }$ i<sub>npu</sub>t <sub>con</sub>t<sub>ex</sub>t<sub>s.</sub> F<sub>or</sub> fi<sub>xe</sub>d �<sub>, an unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d bi<sub>nary or</sub>d<sub>er-</sub>� M<sub>ar</sub>k<sub>ov</sub> f<sub>am</sub>il<sub>y</sub> i<sub>ns</sub>t<sub>ea</sub>d h<sub>as</sub> $2 ^ { k }$ f<sub>ree</sub> t<sub>rans</sub>iti<sub>on pro</sub>b<sub>a</sub>biliti<sub>es an</sub>d<sub>, un</sub>d<sub>er</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or</sub>it<sub>y, re</sub>d<sub>un</sub>d<sub>ancy o</sub>f <sub>or</sub>d<sub>er</sub> $2 ^ { k }$ lo<sub>g</sub> � (Atteson 1999). Thus Markov order can overstate the <sub>p</sub>rediction com<sub>p</sub>lexit<sub>y</sub> ex<sub>p</sub>onentiall<sub>y</sub> even before one considers i<sub>n</sub>fi<sub>n</sub>it<sub>e memory.</sub>

Corollary 4.8 (Exponential envelope). Let

$$
r _ { j } = A e ^ { - \alpha j } , \qquad A > 0 , \quad \alpha > 0 , \qquad \sum _ { j \geq 1 } r _ { j } \leq B .\tag{52}
$$

Write $\Lambda _ { T } = \log ( A ^ { 2 } T )$ . There are constants $c _ { B } , C _ { B } > 0$ such that, whenever

$$
\Lambda _ { T } \ge C _ { B } ( 1 + \alpha ) , \qquad \left( \frac { \Lambda _ { T } } { \alpha } \right) ^ { 2 } \log ( 2 T ) \le c _ { B } T ,\tag{53}
$$

we have

$$
c _ { B } \frac { \Lambda _ { T } ^ { 2 } } { \alpha } \leq \mathcal { R } _ { T } ( r ) \leq C _ { B } \frac { \Lambda _ { T } ^ { 2 } } { \alpha } .\tag{54}
$$

Consequently, for fixed �, � satisfying (52),

$$
\mathcal { R } _ { T } ( r ) = \Theta _ { A , \alpha , B } ( \log ^ { 2 } T ) .\tag{55}
$$

The finite-sample bounds in (54) display the leading scale $\alpha ^ { - 1 } \log ^ { 2 } T$ when � and � are held fixed.

Th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>ac</sub>ti<sub>ve</sub> l<sub>ags</sub> i<sub>s</sub> <sub>on</sub>l<sub>y</sub> $\Theta ( \alpha ^ { - 1 } \log T )$ <sub>,</sub> b<sub>u</sub>t <sub>eac</sub>h <sub>ac</sub>ti<sub>ve</sub> l<sub>a mus</sub>t b<sub>e</sub> l<sub>earne</sub>d t<sub>o a reso</sub>l<sub>u</sub>ti<sub>on</sub> d<sub>epen</sub>di<sub>ng on</sub> it<sub>s amp</sub>lit<sub>u</sub>d<sub>e.</sub> S<sub>umm</sub>i<sub>ng</sub> th<sub>ese reso</sub>l<sub>u</sub>ti<sub>ons pro</sub>d<sub>uces</sub> th<sub>e square</sub>d l<sub>ogar</sub>ith<sub>m.</sub> Th<sub>e resu</sub>lt i<sub>s</sub> <sub>a</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve-regre</sub>t <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>;</sub> it <sub>s</sub>h<sub>ou</sub>ld <sub>no</sub>t b<sub>e</sub> <sub>con</sub>f<sub>use</sub>d <sub>w</sub>ith <sub>one-s</sub>t<sub>ep</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>r</sub>i<sub>s</sub>k <sub>a</sub>ft<sub>er</sub> <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> <sub>a</sub> trajectory.

Corollary 4.9 (Polynomial envelope). Let

$$
r _ { j } = A j ^ { - s } , \qquad A > 0 , \quad s > 1 , \qquad \sum _ { j \geq 1 } r _ { j } \leq B ,\tag{56}
$$

and set $M _ { T } = ( A ^ { 2 } T ) ^ { 1 / ( 2 s ) }$ . There are constants $c _ { B , s } , C _ { B , s } > 0$ such that, whenever

$$
M _ { T } \geq C _ { B , s } , \qquad M _ { T } ^ { 2 } \log ( 2 T ) \leq c _ { B , s } T ,\tag{57}
$$

we have

$$
c _ { B , s } M _ { T } \leq \mathcal { R } _ { T } ( r ) \leq C _ { B , s } M _ { T } .\tag{58}
$$

For fixed � and $s > 1$ , the conditions hold for all suficiently large �, and therefore

$$
\begin{array} { r } { \mathcal { R } _ { T } ( r ) = \Theta _ { A , s , B } \Big ( T ^ { 1 / ( 2 s ) } \Big ) . } \end{array}\tag{59}
$$

Th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f <sub>a</sub>ll th<sub>ree</sub> <sub>coro</sub>ll<sub>ar</sub>i<sub>es</sub> i<sub>s</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> E<sub>.</sub> F<sub>or</sub> th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>case,</sub> th<sub>e</sub> k<sub>ey</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>s</sub>

$$
\sum _ { j \ge 1 } \log \Bigl ( 1 + A ^ { 2 } T j ^ { - 2 s } \Bigr ) \asymp _ { s } ( A ^ { 2 } T ) ^ { 1 / ( 2 s ) } .\tag{60}
$$

A h<sub>ar</sub>d t<sub>runca</sub>ti<sub>on</sub> b<sub>oun</sub>d <sub>o</sub>f th<sub>e</sub> f<sub>orm</sub> ℎ l<sub>og</sub> $T + T A ^ { 2 } h ^ { 1 - 2 s }$ i<sub>ns</sub>t<sub>ea</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>zes</sub> t<sub>o</sub>

$$
T ^ { 1 / ( 2 s ) } ( \log T ) ^ { 1 - 1 / ( 2 s ) } ,\tag{61}
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> l<sub>arger.</sub> Th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d <sub>spec</sub>t<sub>rum</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>no</sub>t <sub>mere</sub>l<sub>y</sub> <sub>a</sub> <sub>re</sub>fi<sub>ne</sub>d <sub>proo</sub>f <sub>o</sub>f th<sub>e</sub> <sub>same</sub> <sub>orac</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y;</sub> it <sub>c</sub>h<sub>anges</sub> th<sub>e ra</sub>t<sub>e.</sub>

Why memory profiles do not determine regret. The regret rates above quantify the cost of learning th<sub>e un</sub>k<sub>nown</sub> l<sub>ag coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub> T<sub>o compare</sub> th<sub>a</sub>t <sub>cos</sub>t <sub>w</sub>ith th<sub>e approx</sub>i<sub>ma</sub>ti<sub>on error cause</sub>d b<sub>y</sub> fi<sub>n</sub>it<sub>e memory,</sub> C<sub>oro</sub>ll<sub>ary</sub> 3<sub>.</sub>5 <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>runca</sub>ti<sub>on</sub> <sub>pro</sub>fil<sub>es.</sub> If $1 \leq h \leq T / 4$ <sub>,</sub> th<sub>en</sub>

$$
\sum _ { j > h } n _ { T , j } A ^ { 2 } e ^ { - 2 \alpha j } \asymp _ { \alpha } T A ^ { 2 } e ^ { - 2 \alpha h } ,\tag{62}
$$

$$
\sum _ { j > h } n _ { T , j } A ^ { 2 } j ^ { - 2 s } \asymp _ { s } T A ^ { 2 } h ^ { 1 - 2 s } .\tag{63}
$$

Th<sub>e exac</sub>t <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e</sub> fi<sub>n</sub>it<sub>e-</sub>h<sub>or</sub>i<sub>zon e</sub>d<sub>ge</sub> t<sub>erms, are g</sub>i<sub>ven</sub> i<sub>n</sub> L<sub>emma</sub> E<sub>.</sub>1<sub>.</sub> Th<sub>ese pro</sub>fil<sub>es quan</sub>tif<sub>y</sub> h<sub>ow muc</sub>h <sub>pre</sub>di<sub>c</sub>ti<sub>ve</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s</sub> l<sub>os</sub>t b<sub>y remov</sub>i<sub>ng o</sub>ld l<sub>ags,</sub> b<sub>u</sub>t <sub>no</sub>t h<sub>ow many un</sub>k<sub>nown</sub> di<sub>rec</sub>ti<sub>ons genera</sub>t<sub>e</sub> th<sub>e re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>ve</sub> l<sub>aw.</sub> T<sub>o</sub> i<sub>so</sub>l<sub>a</sub>t<sub>e</sub> thi<sub>s m</sub>i<sub>ss</sub>i<sub>ng parame</sub>t<sub>er geome</sub>t<sub>ry, cons</sub>id<sub>er</sub> th<sub>e ran</sub>k<sub>-one c</sub>l<sub>ass</sub> $\Theta _ { \mathrm { r a y } } ( r )$ defined in (29). Ever<sub>y</sub> la<sub>g</sub> remains <sub>p</sub>resent with the same deca<sub>y p</sub>rofile, but all coeficients share one scalar <sub>parame</sub>t<sub>er.</sub> Th<sub>us</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>e enve</sub>l<sub>ope</sub> $\Theta ( r )$ <sub>an</sub>d th<sub>e ray</sub> h<sub>ave</sub> th<sub>e same ex</sub>t<sub>rema</sub>l <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>sequence �,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub>i<sub>r num</sub>b<sub>ers o</sub>f i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y un</sub>k<sub>nown</sub> di<sub>rec</sub>ti<sub>ons are en</sub>ti<sub>re</sub>l<sub>y</sub> dif<sub>eren</sub>t<sub>.</sub> L<sub>e</sub>t

$$
I _ { T } ( r ) = \sum _ { j = 1 } ^ { T } n _ { T , j } r _ { j } ^ { 2 } .\tag{64}
$$

Theorem 4.10 (Memory-profile impossibility). For every summable nonincreasing � with $r _ { 1 } > 0 ,$ , the coordinate envelope $\Theta ( r )$ and the ray $\Theta _ { \mathrm { r a y } } ( r )$ have equivalent worst-case truncation bias and worst-case recent-input distortion, up to constants depending only on $B ,$ as quantified by Theorem 3.3. In contrast, the minimax redundancy ofthe ray satisfies

$$
\left[ \frac { 1 } { 2 } \log ( T r _ { 1 } ^ { 2 } ) - C _ { B } \right] _ { + } \leq \mathcal { R } _ { T } ^ { \mathrm { r a y } } ( r ) \leq \frac { 1 } { 2 } \log ( 1 + I _ { T } ( r ) ) + C ,\tag{65}
$$

where $\mathcal { R } _ { T } ^ { \mathrm { r a y } } ( r )$ denotes (17) with the supremum restricted to (29). Hence,for everyfixed profile with $r _ { 1 } > 0 ,$

$$
\mathcal { R } _ { T } ^ { \mathrm { r a y } } ( r ) = \Theta _ { B , r _ { 1 } } ( \log T ) .\tag{66}
$$

In particular, for $r _ { j } = A j ^ { - s }$ with $s > 1$ and $1 \leq h \leq T / 4$

$$
\operatorname* { s u p } _ { \theta \in \Theta ( r ) } \mathcal { D } _ { T } ( \theta , h ) \asymp _ { B } \operatorname* { s u p } _ { \theta \in \Theta _ { \mathrm { r a y } } ( r ) } \mathcal { D } _ { T } ( \theta , h ) \asymp _ { B , s } T A ^ { 2 } h ^ { 1 - 2 s } ,\tag{67}
$$

while

$$
\mathcal { R } _ { T } ( r ) = \Theta \Big ( T ^ { 1 / ( 2 s ) } \Big ) , \qquad \mathcal { R } _ { T } ^ { \mathrm { r a y } } ( r ) = \Theta ( \log T ) .\tag{68}
$$

Proof. The equality of the worst-case memory profiles is Corollary 3.5. The redundancy bounds are <sub>prove</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> G<sub>.</sub> Th<sub>e upper</sub> b<sub>oun</sub>d i<sub>s a one-</sub>di<sub>mens</sub>i<sub>ona</sub>l l<sub>oca</sub>li<sub>ze</sub>d <sub>m</sub>i<sub>x</sub>t<sub>ure w</sub>ith <sub>pre</sub>di<sub>c</sub>t<sub>a</sub>bl<sub>e</sub> f<sub>ea</sub>t<sub>ure</sub> $\begin{array} { r } { z _ { t } = \sum _ { j \leq t } r _ { j } U _ { t + 1 - j } } \end{array}$ <sub>.</sub> F<sub>or</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>, con</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e ear</sub>li<sub>er</sub> i<sub>npu</sub>t<sub>s,</sub> $z _ { t } = w _ { t } + r _ { 1 } U _ { t }$ <sub>an</sub>d <sub>a</sub>t l<sub>eas</sub>t <sub>one o</sub>f $\vert w _ { t } + r _ { 1 } \vert$ <sub>an</sub>d $\vert w _ { t } - r _ { 1 } \vert$ | is at least $r _ { 1 }$ <sub>.</sub> A <sub>mar</sub>ti<sub>nga</sub>l<sub>e argumen</sub>t th<sub>ere</sub>f<sub>ore g</sub>i<sub>ves</sub> li<sub>near emp</sub>i<sub>r</sub>i<sub>ca</sub>l Fi<sub>s</sub>h<sub>er</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> with hi<sub>g</sub>h <sub>p</sub>robabilit<sub>y</sub>. A one-dimensional constrained-MLE entro<sub>py</sub> ar<sub>g</sub>ument then <sub>y</sub>ields the left side of (65). Finall<sub>y</sub>, combine Corollar<sub>y</sub> 4.9 with (66). □

Remark 4.11 (What a valid memory-complexity measure must retain). Theorem 4.10 rules out any minimax <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on</sub> th<sub>a</sub>t d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>sca</sub>l<sub>ar</sub> <sub>map</sub> $h \mapsto \operatorname* { s u p } _ { \theta } \mathcal { D } _ { T } ( \theta , h )$ <sub>.</sub> A <sub>va</sub>lid <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on mus</sub>t <sub>a</sub>l<sub>so</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e me</sub>t<sub>r</sub>i<sub>c or co</sub>di<sub>ng comp</sub>l<sub>ex</sub>it<sub>y o</sub>f th<sub>e pre</sub>di<sub>c</sub>ti<sub>ve</sub> l<sub>aws rea</sub>li<sub>z</sub>i<sub>ng eac</sub>h di<sub>s</sub>t<sub>or</sub>ti<sub>on</sub> l<sub>eve</sub>l<sub>.</sub> I<sub>n</sub> th<sub>e presen</sub>t f<sub>am</sub>il<sub>y,</sub> thi<sub>s</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s</sub> <sub>cap</sub>t<sub>ure</sub>d b<sub>y</sub> th<sub>e</sub> <sub>an</sub>i<sub>so</sub>t<sub>rop</sub>i<sub>c</sub> <sub>re</sub>d<sub>un</sub>d<sub>ancy</sub> <sub>spec</sub>t<sub>rum</sub> $\Gamma _ { T } ( r )$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>s</sub>h<sub>arp</sub> f<sub>or</sub> th<sub>e</sub> canonical envelo<sub>p</sub>es studied here; in a diferent famil<sub>y</sub> it ma<sub>y</sub> be a Shtarkov com<sub>p</sub>lexit<sub>y</sub> (Shtar’kov 1987), a l<sub>oca</sub>l <sub>me</sub>t<sub>r</sub>i<sub>c</sub> <sub>en</sub>t<sub>ropy,</sub> <sub>an</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>ra</sub>di<sub>us,</sub> <sub>or</sub> <sub>ano</sub>th<sub>er</sub> <sub>sequen</sub>ti<sub>a</sub>l <sub>co</sub>di<sub>ng</sub> <sub>quan</sub>tit<sub>y.</sub>

## 5 Computational Predictors Attaining the Redundancy Spectrum

S<sub>ec</sub>ti<sub>on</sub> 4 d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>max regre</sub>t <sub>us</sub>i<sub>ng unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d <sub>causa</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y ass</sub>i<sub>gnmen</sub>t<sub>s.</sub> It<sub>s</sub> B<sub>ayes</sub>i<sub>an</sub> <sub>co</sub>d<sub>e proves s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>ac</sub>hi<sub>eva</sub>bilit<sub>y</sub> b<sub>u</sub>t i<sub>s no</sub>t <sub>presen</sub>t<sub>e</sub>d <sub>as an e</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on.</sub> Thi<sub>s sec</sub>ti<sub>on</sub> h<sub>as</sub> t<sub>wo</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>goa</sub>l<sub>s.</sub> Fi<sub>rs</sub>t<sub>,</sub> it <sub>g</sub>i<sub>ves</sub> <sub>an</sub> <sub>exp</sub>li<sub>c</sub>it <sub>on</sub>li<sub>ne</sub> N<sub>ew</sub>t<sub>on</sub> <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> <sub>a</sub>tt<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> $O ( \Gamma _ { T } ( r ) )$ <sub>.</sub> S<sub>econ</sub>d<sub>,</sub> it t<sub>rea</sub>t<sub>s</sub> <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on over can</sub>did<sub>a</sub>t<sub>e enve</sub>l<sub>opes an</sub>d <sub>compresse</sub>d filt<sub>er</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>ons.</sub>

F<sub>or</sub> th<sub>e</sub> l<sub>agge</sub>d <sub>source,</sub> <sub>we</sub> id<sub>en</sub>tif<sub>y</sub> <sub>a</sub> <sub>mar</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> $\widehat { p _ { t } }$ with the joint next-symbol assignment

$$
q _ { t } ( ( u , y ) \mid X _ { 1 : t } ) = \frac { 1 } { 2 } \widehat { p } _ { t } ^ { y } ( 1 - \widehat { p } _ { t } ) ^ { 1 - y } .
$$

Thus the re<sub>g</sub>ret notation below is the joint-source re<sub>g</sub>ret in (10); the known factor $1 / 2$ f<sub>or</sub> th<sub>e</sub> f<sub>res</sub>h i<sub>npu</sub>t <sub>cance</sub>l<sub>s</sub> f<sub>rom</sub> th<sub>e excess</sub> l<sub>oss.</sub>

## 5.1 Spectrum-attaining online Newton predictor

Fi<sub>x</sub> <sub>an</sub> <sub>enve</sub>l<sub>ope</sub> <sub>�</sub> <sub>an</sub>d <sub>a</sub> h<sub>or</sub>i<sub>zon</sub> �<sub>.</sub> Si<sub>nce</sub> b<sub>o</sub>th $n _ { T , j }$ <sub>an</sub>d $r _ { j }$ <sub>are</sub> <sub>non</sub>i<sub>ncreas</sub>i<sub>ng</sub> i<sub>n</sub> $j ,$ <sub>so</sub> i<sub>s</sub> $s _ { T , j } = n _ { T , j } r _ { j } ^ { 2 }$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> th<sub>e un</sub>it<sub>-s</sub>i<sub>gna</sub>l <sub>cu</sub>t<sub>o</sub>f

$$
h _ { T } ( r ) = \operatorname* { m a x } \{ j \leq T : s _ { T , j } \geq 1 \} ,\tag{69}
$$

<sub>w</sub>ith $h _ { T } ( r ) = 0$ if th<sub>e</sub> <sub>se</sub>t i<sub>s</sub> <sub>emp</sub>t<sub>y.</sub> F<sub>or</sub> $h = h _ { T } ( r )$ <sub>,</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>parame</sub>t<sub>ers</sub> $u _ { j } = \theta _ { j } / r _ { j } \in [ - 1 , 1 ]$ <sub>an</sub>d<sub>,</sub> b<sub>e</sub>f<sub>ore pre</sub>di<sub>c</sub>ti<sub>ng</sub> $Y _ { t + 1 }$ <sub>,</sub> th<sub>e pre</sub>di<sub>c</sub>t<sub>a</sub>bl<sub>e</sub> f<sub>ea</sub>t<sub>ure vec</sub>t<sub>or</sub>

$$
\boldsymbol { x } _ { t } ^ { ( h ) } = \left( r _ { 1 } U _ { t } , r _ { 2 } U _ { t - 1 } , \ldots , r _ { h \wedge t } U _ { t + 1 - h \wedge t } , 0 , \ldots , 0 \right) \in \mathbb { R } ^ { h } .\tag{70}
$$

<sup>E</sup>ver<sub>y</sub> $u \in [ - 1 , 1 ] ^ { h }$ <sub>p</sub>roduces a lo<sub>g</sub>it in [−�, �]. Run Online Newton Ste<sub>p</sub> (ONS) on the lo<sub>g</sub>istic loss

$$
\ell _ { t } ( u ) = \psi ( \langle u , x _ { t } ^ { ( h ) } \rangle ) - Y _ { t + 1 } \langle u , x _ { t } ^ { ( h ) } \rangle .\tag{71}
$$

S<sub>e</sub>t $\beta _ { B } = \kappa _ { B }$ <sub>.</sub> Th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>pre</sub>di<sub>c</sub>t<sub>s</sub> $\sigma ( \langle u _ { t } , x _ { t } ^ { ( h ) } \rangle )$ and projects each Newton update back to the box $[ - 1 , 1 ] ^ { h }$ i<sub>n</sub> th<sub>e</sub> <sub>curren</sub>t <sub>qua</sub>d<sub>ra</sub>ti<sub>c</sub> <sub>me</sub>t<sub>r</sub>i<sub>c.</sub> A <sub>concre</sub>t<sub>e</sub> <sub>vers</sub>i<sub>on</sub> i<sub>s</sub> di<sub>sp</sub>l<sub>aye</sub>d i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 1<sub>;</sub> <sub>any</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t b<sub>oun</sub>d<sub>e</sub>d<sub>-</sub>l<sub>og</sub>it ONS t<sub>u</sub>nin<sub>g g</sub>i<sub>ves</sub> th<sub>e sa</sub>m<sub>e o</sub>rd<sub>e</sub>r<sub>.</sub> If $h = 0 ;$ <sub>,</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m s</sub>i<sub>mp</sub>l<sub>y pre</sub>di<sub>c</sub>t<sub>s</sub> $1 / 2$ f<sub>or every mar</sub>k <sub>an</sub>d <sub>per</sub>f<sub>orms no</sub> Newton u<sub>p</sub>dates. For a <sub>p</sub>ositive-definite matrix � and a closed convex set S, write $\Pi _ { S } ^ { H } ( \nu )$ f<sub>or</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer</sub> <sub>o</sub>f $( u - \nu ) ^ { \top } H ( u - \nu )$ over $u \in S$

Algorithm 1 Profile-scaled ONS for a known envelope   
Require: Envelope $r ,$ h<sub>or</sub>i<sub>zon</sub> �<sub>,</sub> l<sub>og</sub>it b<sub>oun</sub>d �   
1<sub>:</sub> S<sub>e</sub>t $\beta _ { B } = \kappa _ { B } = \sigma ( B ) ( 1 - \sigma ( B ) )$   
2<sub>:</sub> S<sub>e</sub>t $h = h _ { T } ( r ) , u _ { 1 } = 0 \in \mathbb { R } ^ { h } .$ <sub>,</sub> <sub>an</sub>d $H _ { 0 } = I _ { h }$   
3<sub>:</sub> Ob<sub>serve</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>sym</sub>b<sub>o</sub>l $X _ { 1 } = ( U _ { 1 } , Y _ { 1 } )$   
4: for $t = 1 , \dots , T$ do   
5<sub>:</sub> F<sub>orm</sub> $x _ { t } ^ { ( h ) }$ f<sub>rom</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d i<sub>npu</sub>t<sub>s</sub> $U _ { 1 : t }$ usin<sub>g</sub> (70)   
6<sub>:</sub> P<sub>re</sub>di<sub>c</sub>t $\widehat { p _ { t } } = \sigma ( \langle u _ { t } , x _ { t } ^ { ( h ) } \rangle )$ f<sub>or</sub> $Y _ { t + 1 }$ <sub>an</sub>d <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $1 / 2$ f<sub>or</sub> $U _ { t + 1 }$   
7<sub>:</sub> Ob<sub>serve</sub> $X _ { t + 1 } = ( U _ { t + 1 } , Y _ { t + 1 } )$ <sub>an</sub>d <sub>se</sub>t $g _ { t } = ( \widehat { p _ { t } } - Y _ { t + 1 } ) x _ { t } ^ { ( h ) }$   
8<sub>:</sub> $H _ { t } = H _ { t - 1 } + \beta _ { B } g _ { t } g _ { t } ^ { \top }$   
9: $u _ { t + 1 } = \Pi _ { [ - 1 , 1 ] ^ { h } } ^ { H _ { t } } \big ( u _ { t } - H _ { t } ^ { - 1 } g _ { t } \big )$

Theorem 5.1 (A spectrum-attaining online Newton predictor). There is a constant $C _ { B } > 0 ,$ , depending only on �, such that Algorithm 1 with $\beta _ { B } = \kappa _ { B }$ satisfies, for every $\theta \in \Theta ( r )$

$$
\mathrm { R e g } _ { T } ( \mathrm { O N S } _ { r } ; P _ { \theta , T } ) \leq C _ { B } \left[ h + \sum _ { j = 1 } ^ { h } \log ( 1 + n _ { T , j } r _ { j } ^ { 2 } ) \right] + \frac { 1 } { 8 } \sum _ { j > h } n _ { T , j } \theta _ { j } ^ { 2 }\tag{72}
$$

$$
\le C _ { B } \Gamma _ { T } ( r ) , \qquad h = h _ { T } ( r ) .\tag{73}
$$

The Newton matrix $H _ { t }$ and its inverse can be maintained in $O ( h ^ { 2 } )$ arithmetic operations per round and $O ( h ^ { 2 } )$ memory by a rank-one update. Each round additionally requires the convex quadratic projection onto $[ - 1 , 1 ] ^ { h }$ in the $H _ { t }$ metric; this is polynomial-time, and its exact cost depends on the projection solver.

Proof. The oracle inequality (72) is proved in Appendix F. ONS gives the pathwise comparison bound

$$
\sum _ { t = 1 } ^ { T } \ell _ { t } ( u _ { t } ) - \ell _ { t } ( u ) \leq C _ { B } \left[ h + \log \operatorname* { d e t } \biggl ( I _ { h } + \sum _ { t = 1 } ^ { T } x _ { t } ^ { ( h ) } ( x _ { t } ^ { ( h ) } ) ^ { \top } \biggr ) \right] .\tag{74}
$$

H<sub>a</sub>d<sub>amar</sub>d’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> b<sub>oun</sub>d<sub>s</sub> th<sub>e</sub> l<sub>og</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nan</sub>t b<sub>y</sub> $\begin{array} { r } { \sum _ { j \le h } \log ( 1 + n _ { T , j } r _ { j } ^ { 2 } ) } \end{array}$ <sub>.</sub> C<sub>om are w</sub>ith th<sub>e norma</sub>li<sub>ze</sub>d t<sub>runca</sub>ti<sub>on</sub> $u _ { j } = \theta _ { j } / r _ { j }$ <sub>an</sub>d i<sub>nvo</sub>k<sub>e</sub> Th<sub>eorem</sub> 3<sub>.</sub>3 f<sub>or</sub> th<sub>e om</sub>itt<sub>e</sub>d t<sub>a</sub>il<sub>.</sub>

To derive (73) from (72), note that ever<sub>y</sub> $j \leq h$ h<sub>as</sub> $s _ { T , j } \geq 1$ <sub>,</sub> h<sub>ence</sub> $h \leq \Gamma _ { T } ( r ) / \log 2$ . <sup>E</sup>ver<sub>y</sub> $j > h$ h<sub>as</sub> $s _ { T , j } < 1$ <sub>, an</sub>d $s _ { T , j } \leq 2 \log ( 1 + s _ { T , j } )$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub> b<sub>o</sub>th th<sub>e</sub> di<sub>mens</sub>i<sub>on</sub> t<sub>erm an</sub>d th<sub>e wors</sub>t<sub>-case</sub> t<sub>a</sub>il t<sub>erm are</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y a cons</sub>t<sub>an</sub>t <sub>mu</sub>lti<sub>p</sub>l<sub>e o</sub>f $\Gamma _ { T } ( r )$ □

Remark 5.2 (Horizon dependence). The cutof in (69) uses the horizon. The Bayesian mixture in Theorem 4.1, i<sub>mp</sub>l<sub>emen</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>e</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>e pro</sub>d<sub>uc</sub>t <sub>pr</sub>i<sub>or</sub> d<sub>escr</sub>ib<sub>e</sub>d th<sub>ere,</sub> i<sub>s any</sub>ti<sub>me,</sub> b<sub>u</sub>t th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d fi<sub>n</sub>it<sub>e-</sub>di<sub>mens</sub>i<sub>ona</sub>l ONS i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d f<sub>or</sub> k<sub>nown</sub> �<sub>.</sub> N<sub>a</sub>i<sub>ve</sub> <sub>geome</sub>t<sub>r</sub>i<sub>c</sub> <sub>res</sub>t<sub>ar</sub>ti<sub>ng</sub> <sub>can</sub> <sub>repea</sub>t<sub>e</sub>dl<sub>y</sub> <sub>repay</sub> th<sub>e</sub> l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> <sub>reso</sub>l<sub>u</sub>ti<sub>on</sub> <sub>cos</sub>t <sub>o</sub>f <sub>an</sub> <sub>ac</sub>ti<sub>ve</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> <sub>nee</sub>d <sub>no</sub>t <sub>preserve</sub> $\Gamma _ { T } ( r )$ <sub>w</sub>ithi<sub>n</sub> <sub>a</sub> <sub>cons</sub>t<sub>an</sub>t f<sub>ac</sub>t<sub>or.</sub> A h<sub>or</sub>i<sub>zon-</sub>f<sub>ree</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>can</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>aggrega</sub>t<sub>e</sub> fi<sub>xe</sub>d<sub>-cu</sub>t<sub>o</sub>f l<sub>earners,</sub> <sub>a</sub>t th<sub>e</sub> <sub>pr</sub>i<sub>ce</sub> <sub>o</sub>f th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>pr</sub>i<sub>or</sub> <sub>pena</sub>lt<sub>y,</sub> <sub>or</sub> <sub>use</sub> <sub>a</sub> ti<sub>me-vary</sub>i<sub>ng</sub> <sub>regu</sub>l<sub>ar</sub>i<sub>zer;</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> <sub>c</sub>l<sub>eanes</sub>t <sub>any</sub>ti<sub>me</sub> <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> b<sub>oun</sub>d i<sub>s separa</sub>t<sub>e</sub> f<sub>rom</sub> th<sub>e s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>resu</sub>lt<sub>s</sub> h<sub>ere.</sub>

## 5.2 Adaptation and compressed implementations

Adaptation to an unknown profile. Suppose the learner is given a countable collection of candidate enve<sup>l</sup>o<sub>p</sub>es $\{ r ^ { ( k ) } : k \in \mathcal { K } \}$ <sub>,</sub> <sub>a</sub>ll <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> <sub>a</sub> <sub>common</sub> $\ell _ { 1 }$ b<sub>oun</sub>d �<sub>,</sub> <sub>an</sub>d <sub>pr</sub>i<sub>or</sub> <sub>masses</sub> $\pi _ { k } > 0$ <sub>w</sub>ith $\textstyle \sum _ { k } \pi _ { k } = 1$ R<sub>un</sub> th<sub>e pro</sub>fil<sub>e-sca</sub>l<sub>e</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> f<sub>or eac</sub>h <sub>can</sub>did<sub>a</sub>t<sub>e an</sub>d <sub>com</sub>bi<sub>ne</sub> th<sub>e</sub>i<sub>r</sub> B<sub>ernou</sub>lli <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> b<sub>y</sub> th<sub>e</sub> l<sub>og-</sub>l<sub>oss</sub> <sub>m</sub>i<sub>x</sub>t<sub>ure.</sub> F<sub>or any sequence o</sub>f <sub>mar</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>ons</sub> $p = \left( p _ { t } \right)$ <sub>, wr</sub>it<sub>e</sub>

$$
L _ { t } ( p ) = \sum _ { s = 1 } ^ { t } [ - Y _ { s + 1 } \log p _ { s } - ( 1 - Y _ { s + 1 } ) \log ( 1 - p _ { s } ) ] .
$$

S<sub>e</sub>t $L _ { 0 } ( p ) = 0$ <sub>.</sub> F<sub>or</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> thi<sub>s</sub> i<sub>s</sub> <sub>a</sub> di<sub>rec</sub>t <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>proce</sub>d<sub>ure;</sub> f<sub>or</sub> <sub>a</sub> <sub>genu</sub>i<sub>ne</sub>l<sub>y</sub> <sub>coun</sub>t<sub>a</sub>bl<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>emen</sub>t i<sub>s</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on-</sub>th<sub>eore</sub>ti<sub>c un</sub>l<sub>ess</sub> th<sub>e pr</sub>i<sub>or an</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e</sub> f<sub>am</sub>il<sub>y a</sub>d<sub>m</sub>it <sub>a</sub> l<sub>azy or</sub> t<sub>runca</sub>t<sub>e</sub>d i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on.</sub> L<sub>e</sub>t $L _ { t - 1 , k } = L _ { t - 1 } ( \widehat { p } _ { k } )$ <sub>an</sub>d <sub>se</sub>t

$$
w _ { t , k } = \frac { \pi _ { k } e ^ { - L _ { t - 1 , k } } } { \sum _ { \ell \in \mathcal { K } } \pi _ { \ell } e ^ { - L _ { t - 1 , \ell } } } , \qquad \widehat { p } _ { t } = \sum _ { k \in \mathcal { K } } w _ { t , k } \widehat { p } _ { t , k } .\tag{75}
$$

Corollary 5.3 (Profile adaptation). The mixture (75) satisfies, simultaneously for every � and every $\theta \in \Theta ( r ^ { ( k ) } )$ ,

$$
\mathrm { R e g } _ { T } ( \widehat { p } ; P _ { \theta , T } ) \leq C _ { B } \Gamma _ { T } ( r ^ { ( k ) } ) + \log \frac { 1 } { \pi _ { k } } .\tag{76}
$$

Consequently, a finite or implementable countable grid over memory cutofs or decay parameters adapts whenever one candidate contains the true coeficient sequence (or a dominating envelope), with the explicit additional cost log $( 1 / \pi _ { k } )$ . For example, a polynomially decaying prior on an integer grid index gives an $O ( \log k )$ selection penalty.

Proof. For log loss, a probability mixture satisfies $L _ { T } ( \widehat { p } ) \leq L _ { T } ( \widehat { p } _ { k } ) + \log ( 1 / \pi _ { k } )$ <sub>pa</sub>th<sub>w</sub>i<sub>se.</sub> A<sub>pp</sub>l<sub>y</sub> Th<sub>eorem</sub> 5<sub>.</sub>1 t<sub>o can</sub>did<sub>a</sub>t<sub>e</sub> �<sub>.</sub> □

Predictable-design extension. The lagged Rademacher model is essential for the matching minimax lower b<sub>oun</sub>d<sub>,</sub> b<sub>u</sub>t th<sub>e a</sub>l<sub>gor</sub>ith<sub>m</sub>i<sub>c upper</sub> b<sub>oun</sub>d <sub>requ</sub>i<sub>res ne</sub>ith<sub>er</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t i<sub>npu</sub>t<sub>s nor</sub> T<sub>oep</sub>lit<sub>z s</sub>t<sub>ruc</sub>t<sub>ure.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t <sub>separa</sub>t<sub>es</sub> th<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> <sub>on</sub>li<sub>ne-</sub>l<sub>earn</sub>i<sub>ng</sub> <sub>par</sub>t f<sub>rom</sub> th<sub>e</sub> <sub>source-spec</sub>ifi<sub>c</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on.</sub>

L<sub>e</sub>t � b<sub>e a</sub> l<sub>aw un</sub>d<sub>er w</sub>hi<sub>c</sub>h $\phi _ { t } \ \in \mathbb { R } ^ { d }$ i<sub>s measura</sub>bl<sub>e</sub> b<sub>e</sub>f<sub>ore</sub> $Y _ { t + 1 }$ i<sub>s revea</sub>l<sub>e</sub>d <sub>an</sub>d th<sub>e</sub> t<sub>rue con</sub>diti<sub>ona</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s</sub> $\sigma ( \eta _ { t + 1 } ^ { \star } )$ <sub>w</sub>ith $| \eta _ { t + 1 } ^ { \star } | \leq B$ <sub>.</sub> F<sub>or</sub> <sub>a</sub> <sub>c</sub>l<sub>ose</sub>d <sub>convex</sub> <sub>parame</sub>t<sub>er</sub> <sub>se</sub>t $\mathcal { U } \subset \mathbb { R } ^ { d }$ <sub>,</sub> l<sub>e</sub>t th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> l<sub>og</sub>it<sub>s</sub> b<sub>e</sub> $\eta _ { t + 1 } ( u ) = \langle u , \phi _ { t } \rangle$ <sub>, a</sub>l<sub>so</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> � f<sub>or</sub> $u \in { \mathcal { U } } .$ <sub>.</sub> R<sub>un</sub> ONS <sub>on</sub> th<sub>e</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> l<sub>oss w</sub>ith $u _ { 1 } \in \mathcal { U } , H _ { 0 } = I _ { d }$ <sub>,</sub> <sub>an</sub>d $\beta _ { B } = \kappa _ { B }$ <sub>.</sub> F<sub>or</sub> thi<sub>s genera</sub>l <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>,</sub> $\mathrm { R e g } _ { T } ( \mathrm { O N S } ; P )$ d<sub>eno</sub>t<sub>es expec</sub>t<sub>e</sub>d <sub>cumu</sub>l<sub>a</sub>ti<sub>ve excess</sub> B<sub>ernou</sub>lli l<sub>og</sub> l<sub>oss</sub> f<sub>or</sub> th<sub>e mar</sub>k<sub>s.</sub>

Theorem 5.4 (Predictable-design oracle inequality). The ONS predictor satisfies

$$
\begin{array} { r l r } {  { \operatorname { R e g } _ { T } ( \operatorname { O N S } ; P ) \le \frac { 1 } { 2 \kappa _ { B } } \mathbb { E } \log \operatorname* { d e t } ( I _ { d } + \sum _ { t = 1 } ^ { T } \phi _ { t } \phi _ { t } ^ { \top } ) } } \\ & { } & { + \ \operatorname* { i n f } _ { u \in \mathcal { U } } \{ \displaystyle \frac { 1 } { 2 } \| u - u _ { 1 } \| _ { 2 } ^ { 2 } + \frac { 1 } { 8 } \sum _ { t = 1 } ^ { T } \mathbb { E } \big ( \eta _ { t + 1 } ^ { \star } - \langle u , \phi _ { t } \rangle \big ) ^ { 2 } \} . } \end{array}\tag{77}
$$

The feature sequence may be stochastic, adaptive, and history-dependent.

Proof. The ONS guarantee is deterministic conditional on the realized features and labels. Compare it with <sub>any</sub> fi<sub>xe</sub>d $u \in \mathcal { U }$ <sub>.</sub> Th<sub>e excess</sub> B<sub>ayes</sub> l<sub>oss o</sub>f th<sub>a</sub>t <sub>compara</sub>t<sub>or</sub> i<sub>s</sub> th<sub>e</sub> B<sub>ernou</sub>lli l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> B<sub>regman</sub> di<sub>vergence</sub> b<sub>e</sub>t<sub>ween</sub> l<sub>og</sub>it<sub>s</sub> $\eta _ { t + 1 } ^ { \star }$ <sub>an</sub>d $\left. u , \phi _ { t } \right.$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s a</sub>t <sub>mos</sub>t <sub>one e</sub>i<sub>g</sub>hth <sub>o</sub>f th<sub>e</sub>i<sub>r square</sub>d dif<sub>erence.</sub> T<sub>a</sub>ki<sub>ng expec</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>a</sub>l<sub>so p</sub>l<sub>aces an expec</sub>t<sub>a</sub>ti<sub>on aroun</sub>d th<sub>e ran</sub>d<sub>om</sub> l<sub>og</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nan</sub>t<sub>.</sub> Fi<sub>na</sub>ll<sub>y op</sub>ti<sub>m</sub>i<sub>ze over �.</sub> □

Compressed filter dictionaries. The coordinate basis is natural for the full envelope because the coeficients <sub>can vary</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y.</sub> Wh<sub>en</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>sequence</sub> h<sub>as a</sub>dditi<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure, a</sub> filt<sub>er</sub> di<sub>c</sub>ti<sub>onary can</sub> b<sub>e</sub> <sub>muc</sub>h <sub>sma</sub>ll<sub>er.</sub> L<sub>e</sub>t $g _ { 1 } , \ldots , g _ { K }$ b<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> k<sub>erne</sub>l<sub>s w</sub>ith $g _ { k , j }$ th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>o</sub>f l<sub>ag</sub> $j ,$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e causa</sub>l filt<sub>er s</sub>t<sub>a</sub>t<sub>e</sub>

$$
z _ { t , k } = \sum _ { j = 1 } ^ { t } g _ { k , j } U _ { t + 1 - j } , \qquad z _ { t } = ( z _ { t , 1 } , \dots , z _ { t , K } ) .\tag{78}
$$

F<sub>or</sub> $a \in \mathbb { R } ^ { K }$ <sub>,</sub> th<sub>e</sub> <sub>rea</sub>d<sub>ou</sub>t $a ^ { \top } z _ { t }$ <sub>correspon</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>sequence</sub> $\begin{array} { r } { ( G a ) _ { j } = \sum _ { k = 1 } ^ { K } a _ { k } g _ { k , j } } \end{array}$

Theorem 5.5 (Low-rank filter approximation). Let $\mathcal { A } \subset \mathbb { R } ^ { K }$ be closed and convex, initialize at $a _ { 1 } \in { \mathcal { A } }$ , and suppose $| a ^ { \top } z _ { t } | \leq B$ for every $a \in { \mathcal { A } }$ , every history, and every �. Then an ONS readout over (78) satisfies,for every $\theta \in \Theta ( r )$

$$
\begin{array} { r l r } {  { \mathrm { R e g } _ { T } ( \mathrm { F i l t e r O N S } ; P _ { \theta , T } ) \le \frac { 1 } { 2 \kappa _ { B } } \mathbb { E } \log \operatorname* { d e t } ( I _ { K } + \sum _ { t = 1 } ^ { T } z _ { t } z _ { t } ^ { \top } ) } } \\ & { } & { + \displaystyle \operatorname* { i n f } _ { a \in \mathcal { A } } \{ \frac { 1 } { 2 } \| a - a _ { 1 } \| _ { 2 } ^ { 2 } + \frac { 1 } { 8 } \sum _ { j = 1 } ^ { T } n _ { T , j } \big ( \theta _ { j } - ( G a ) _ { j } \big ) ^ { 2 } \} . } \end{array}\tag{79}
$$

$I f \| z _ { t } \| _ { 2 } \leq G _ { z }$ deterministically, the log-determinant term is at most

$$
C _ { B } K \log \left( 1 + \frac { T G _ { z } ^ { 2 } } { K } \right) .\tag{80}
$$

Proof. Apply Theorem 5.4 with $\phi _ { t } = z _ { t }$ <sub>.</sub> I<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>an</sub>d <sub>cen</sub>t<sub>er</sub>i<sub>ng</sub> <sub>o</sub>f th<sub>e</sub> R<sub>a</sub>d<sub>emac</sub>h<sub>er</sub> i<sub>npu</sub>t<sub>s</sub> <sub>g</sub>i<sub>ve,</sub> f<sub>or</sub> <sub>every</sub> fi<sub>xe</sub>d $^ { a , }$

$$
\begin{array} { r } { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb E \big ( \eta _ { t + 1 } ( \theta ) - a ^ { \top } z _ { t } \big ) ^ { 2 } = \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { t } \big ( \theta _ { j } - ( G a ) _ { j } \big ) ^ { 2 } } \\ { = \displaystyle \sum _ { j = 1 } ^ { T } n _ { T , j } \big ( \theta _ { j } - ( G a ) _ { j } \big ) ^ { 2 } . } \end{array}
$$

For (80), use lo<sub>g</sub> det $( I + M ) \le K \log ( 1 + \operatorname { t r } ( M ) / K )$ <sub>an</sub>d $\begin{array} { r } { \mathrm { t r } \big ( \sum _ { t } z _ { t } z _ { t } ^ { \top } \big ) \leq T G _ { z } ^ { 2 } } \end{array}$

A <sub>use</sub>f<sub>u</sub>l <sub>spec</sub>i<sub>a</sub>l <sub>case</sub> i<sub>s</sub> th<sub>e</sub> <sub>exponen</sub>ti<sub>a</sub>l di<sub>c</sub>ti<sub>onary</sub> $g _ { k , j } = \lambda _ { k } ^ { j - 1 } , 0 \leq \lambda _ { k } < 1$ <sub>.</sub> It<sub>s</sub> f<sub>ea</sub>t<sub>ures o</sub>b<sub>ey</sub> th<sub>e sca</sub>l<sub>ar</sub> <sup>state-s</sup>p<sup>ace</sup> <sup>rec</sup>u<sup>rrence</sup>

$$
z _ { t + 1 , k } = \lambda _ { k } z _ { t , k } + U _ { t + 1 } , \qquad z _ { 0 , k } = 0 ,\tag{81}
$$

<sub>so</sub> th<sub>e</sub> filt<sub>er</sub> b<sub>an</sub>k i<sub>s up</sub>d<sub>a</sub>t<sub>e</sub>d i<sub>n</sub> $O ( K )$ ti<sub>me an</sub>d $O ( K )$ <sub>s</sub>t<sub>a</sub>t<sub>e memory</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e rea</sub>d<sub>ou</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e.</sub> B<sub>oun</sub>d<sub>s on</sub> a<sub>pp</sub>roximation b<sub>y</sub> ex<sub>p</sub>onential sums, such as those develo<sub>p</sub>ed b<sub>y</sub> Be<sub>y</sub>lkin and Monzon (´ 2010), can therefore be inserted directl<sub>y</sub> into (79). This is the <sub>p</sub>recise role of a state-s<sub>p</sub>ace realization in the <sub>p</sub>resent <sub>p</sub>a<sub>p</sub>er: it is a <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f <sub>a</sub> l<sub>ow-ran</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>ve</sub> di<sub>c</sub>ti<sub>onary,</sub> <sub>no</sub>t th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f <sub>pre</sub>di<sub>c</sub>ti<sub>ve</sub> <sub>memory</sub> <sub>an</sub>d <sub>no</sub>t <sub>an assump</sub>ti<sub>on</sub> th<sub>a</sub>t th<sub>e source</sub> it<sub>se</sub>lf i<sub>s a</sub> li<sub>near</sub> d<sub>ynam</sub>i<sub>ca</sub>l <sub>sys</sub>t<sub>em.</sub>

## 6 Conclusion

F<sub>or</sub> th<sub>e exogenous</sub>l<sub>y</sub> d<sub>r</sub>i<sub>ven</sub> l<sub>agge</sub>d l<sub>og</sub>i<sub>s</sub>ti<sub>c c</sub>l<sub>ass,</sub> $\mathcal { R } _ { T } ( r ) \leq C \Gamma _ { T } ( r )$ f<sub>or every summa</sub>bl<sub>e enve</sub>l<sub>ope an</sub>d $\mathcal { R } _ { T } ( r ) \asymp \Gamma _ { T } ( r )$ f<sub>or</sub> th<sub>e</sub> <sub>canon</sub>i<sub>ca</sub>l <sub>exponen</sub>ti<sub>a</sub>l <sub>an</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enve</sub>l<sub>opes</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>con</sub>diti<sub>ons,</sub> <sub>w</sub>ith <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> d<sub>epen</sub>di<sub>ng on</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>parame</sub>t<sub>ers.</sub> Th<sub>us</sub> th<sub>e respec</sub>ti<sub>ve regre</sub>t <sub>ra</sub>t<sub>es are</sub> $\Theta ( \alpha ^ { - 1 } \log ^ { 2 } T )$ <sub>an</sub>d $\Theta ( T ^ { 1 / ( 2 s ) } )$ <sub>, w</sub>hil<sub>e a memory pro</sub>fil<sub>e a</sub>l<sub>one canno</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>ne regre</sub>t<sub>.</sub> A <sub>sca</sub>l<sub>e</sub>d <sub>on</sub>li<sub>ne</sub> N<sub>ew</sub>t<sub>on pre</sub>di<sub>c</sub>t<sub>or a</sub>tt<sub>a</sub>i<sub>ns</sub> t<sup>h</sup>e s<sub>p</sub>ectrum u<sub>pp</sub>er <sup>b</sup>oun<sup>d</sup>.

## Appendix A Proofs of Proposition 3.1, Lemma 3.2, and Theorem 3.3

Thi<sub>s</sub> <sub>appen</sub>di<sub>x</sub> <sub>proves</sub> th<sub>e</sub> <sub>regre</sub>t id<sub>en</sub>tit<sub>y,</sub> th<sub>e</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> KL <sub>geome</sub>t<sub>ry,</sub> <sub>an</sub>d th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>ve-memory</sub> th<sub>eorem</sub> f<sub>rom</sub> S<sub>ec</sub>ti<sub>o</sub>n 3<sub>.</sub>

## A.1 Proof of Proposition 3.1: Bayes decomposition and redundancy identity

F<sub>or</sub> <sub>every</sub> hi<sub>s</sub>t<sub>ory</sub> $X _ { 1 : t }$ <sub>,</sub> <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r l } { \left. { \mathbb { E } _ { P } \left[ - \log q _ { t } ( X _ { t + 1 } \ | \ X _ { 1 : t } ) + \log p _ { t } ^ { P } ( X _ { t + 1 } \ | \ X _ { 1 : t } ) \right| X _ { 1 : t }  } } \\ & \right]{ = \displaystyle \sum _ { x } p _ { t } ^ { P } ( x \mid X _ { 1 : t } ) \log \frac { p _ { t } ^ { P } ( x \mid X _ { 1 : t } ) } { q _ { t } ( x \mid X _ { 1 : t } ) } } \\ & { = \mathrm { K L } \Big ( p _ { t } ^ { P } ( \cdot \mid X _ { 1 : t } ) \Big \Vert q _ { t } ( \cdot \mid X _ { 1 : t } ) \Big ) . } \end{array}
$$

Takin<sub>g</sub> ex<sub>p</sub>ectations and summin<sub>g</sub> <sub>p</sub>roves the first equalit<sub>y</sub> in (11). The chain rule for relative entro<sub>py</sub> <sub>g</sub>ives

$$
\begin{array} { r l } { \displaystyle \mathrm { K L } ( P | | Q ) = \mathbb { E } _ { P } \log \frac { P ( X _ { 1 : T + 1 } ) } { Q ( X _ { 1 : T + 1 } ) } ~ } & { } \\ { = \mathbb { E } _ { P } \sum _ { t = 1 } ^ { T } \log \frac { p _ { t } ^ { P } ( X _ { t + 1 } \mid X _ { 1 : t } ) } { q _ { t } ( X _ { t + 1 } \mid X _ { 1 : t } ) } , } \end{array}
$$

where the common initial distributions cancel b<sub>y</sub> convention. This is exactl<sub>y</sub> (10). Takin<sub>g</sub> the infimum over � and the su<sub>p</sub>remum over a source class with this common initial law <sub>p</sub>roves (12).

## A.2 Logistic Bregman bounds used in Lemma 3.2

L<sub>e</sub>t $\psi ( z ) = \log ( 1 + e ^ { z } )$ <sub>.</sub> F<sub>or</sub> $p = \sigma ( a )$ <sub>an</sub>d $q = \sigma ( b )$ <sub>,</sub> di<sub>rec</sub>t <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>g</sub>i<sub>ves</sub>

$$
\operatorname { K L } ( \operatorname { B e r } ( p ) \| \operatorname { B e r } ( q ) ) = \psi ( b ) - \psi ( a ) - \psi ^ { \prime } ( a ) ( b - a ) .\tag{82}
$$

B<sub>y</sub> T<sub>ay</sub>l<sub>or</sub>’<sub>s</sub> th<sub>eorem</sub> <sub>w</sub>ith i<sub>n</sub>t<sub>egra</sub>l <sub>rema</sub>i<sub>n</sub>d<sub>er,</sub>

$$
\psi ( b ) - \psi ( a ) - \psi ^ { \prime } ( a ) ( b - a ) = ( b - a ) ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - s ) \psi ^ { \prime \prime } ( a + s ( b - a ) ) d s .\tag{83}
$$

Si<sub>nce</sub>

$$
\psi ^ { \prime \prime } ( z ) = \sigma ( z ) ( 1 - \sigma ( z ) ) \leq \frac { 1 } { 4 }\tag{84}
$$

<sup>f</sup>or ever<sub>y</sub> $z ,$ <sub>an</sub>d $\psi ^ { \prime \prime } ( z ) \geq \kappa _ { B }$ f<sub>or</sub> $z \in [ - B , B ]$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\frac { \kappa _ { B } } { 2 } ( a - b ) ^ { 2 } \leq \mathrm { K L } ( \mathrm { B e r } ( \sigma ( a ) ) \| \mathrm { B e r } ( \sigma ( b ) ) ) \leq \frac { 1 } { 8 } ( a - b ) ^ { 2 }\tag{85}
$$

<sub>w</sub>h<sub>enever</sub> $a , b \in [ - B , B ]$ <sub>.</sub> Th<sub>e upper</sub> b<sub>oun</sub>d i<sub>s g</sub>l<sub>o</sub>b<sub>a</sub>l<sub>.</sub>

## A.3 Proof of Lemma 3.2: Pairwise KL geometry

Th<sub>e</sub> i<sub>npu</sub>t <sub>s</sub>t<sub>ream</sub> h<sub>as</sub> th<sub>e same</sub> l<sub>aw un</sub>d<sub>er</sub> $P _ { \theta , T }$ <sub>an</sub>d $P _ { \theta ^ { \prime } , T }$ <sub>.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> i<sub>npu</sub>t<sub>s,</sub> th<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>,</sub> <sub>an</sub>d th<sub>e c</sub>h<sub>a</sub>i<sub>n ru</sub>l<sub>e g</sub>i<sub>ves</sub>

$$
\mathrm { K L } ( P _ { \theta , T } \Vert P _ { \theta ^ { \prime } , T } ) = \sum _ { t = 1 } ^ { T } \mathbb { E } \mathrm { K L } ( \mathrm { B e r } ( \sigma ( \eta _ { t + 1 } ( \theta ) ) ) \Vert \mathrm { B e r } ( \sigma ( \eta _ { t + 1 } ( \theta ^ { \prime } ) ) ) ) .\tag{86}
$$

S<sub>e</sub>t $\Delta _ { j } = \theta _ { j } - \theta _ { i } ^ { \prime }$ . B<sub>y</sub> (85), each summand is between $\kappa _ { B } / 2$ <sub>an</sub>d $1 / 8$ ti<sub>mes</sub>

$$
\mathbb { E } \left( \sum _ { j = 1 } ^ { t } \Delta _ { j } U _ { t + 1 - j } \right) ^ { 2 } .\tag{87}
$$

The in<sub>p</sub>uts are inde<sub>p</sub>endent, centered, and have unit variance, so the cross terms vanish and (87) equals $\textstyle \sum _ { j = 1 } ^ { t } \Delta _ { j } ^ { 2 }$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
\begin{array} { r } { \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { t } \Delta _ { j } ^ { 2 } = \sum _ { j = 1 } ^ { T } ( T - j + 1 ) \Delta _ { j } ^ { 2 } } \\ { = \displaystyle \sum _ { j = 1 } ^ { T } n _ { T , j } ( \theta _ { j } - \theta _ { j } ^ { \prime } ) ^ { 2 } . } \end{array}
$$

Substitution into (86) <sub>p</sub>roves (22).

## A.4 Proof of Theorem 3.3: Predictive-memory bounds

ProofofTheorem 3.3. Equation (27) follows from Lemma 3.2 with $\theta ^ { \prime } = \theta ^ { ( h ) }$

For (28), fix a round � and write

$$
R _ { t , h } = \sum _ { j \leq h \land t } \theta _ { j } U _ { t + 1 - j } , \qquad W _ { t , h } = \sum _ { h < j \leq t } \theta _ { j } U _ { t + 1 - j } .
$$

Th<sub>e</sub> t<sub>erm</sub> $R _ { t , h }$ i<sub>s</sub> th<sub>e re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f th<sub>e mos</sub>t <sub>recen</sub>t i<sub>npu</sub>t<sub>s, w</sub>hil<sub>e</sub> $W _ { t , h }$ i<sub>s</sub> th<sub>e om</sub>itt<sub>e</sub>d <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> $\mathcal { G } _ { t , h } , R _ { t , h }$ i<sub>s</sub> fi<sub>xe</sub>d <sub>an</sub>d $W _ { t , h }$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>, cen</sub>t<sub>ere</sub>d<sub>, an</sub>d h<sub>as var</sub>i<sub>ance</sub>

$$
\mathbb { E } [ W _ { t , h } ^ { 2 } \mid \mathcal { G } _ { t , h } ] = \sum _ { h < j \leq t } \theta _ { j } ^ { 2 } .
$$

The o<sub>p</sub>timal <sub>p</sub>robabilit<sub>y</sub> in (25) is ${ p } _ { \theta , t } ^ { \star , h } = \mathbb { E } [ \sigma ( R _ { t , h } + W _ { t , h } ) \mid \mathcal { G } _ { t , h } ]$

F<sub>or</sub> th<sub>e</sub> <sub>upper</sub> b<sub>oun</sub>d<sub>,</sub> <sub>compare</sub> <sub>w</sub>ith th<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> $\sigma ( R _ { t , h } )$ <sub>.</sub> Th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>curva</sub>t<sub>ure</sub> b<sub>oun</sub>d $\psi ^ { \prime \prime } \leq 1 / 4$ <sub>g</sub><sup>i</sup>ves

$$
\mathrm { K L } \big ( \mathrm { B e r } ( \sigma ( R _ { t , h } + W _ { t , h } ) ) \big | \big | \mathrm { B e r } ( \sigma ( R _ { t , h } ) ) \big ) \leq \frac { 1 } { 8 } W _ { t , h } ^ { 2 } .
$$

C<sub>on</sub>diti<sub>ona</sub>l B<sub>ayes</sub> <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> <sub>can</sub> <sub>on</sub>l<sub>y</sub> <sub>re</sub>d<sub>uce</sub> th<sub>e</sub> l<sub>oss.</sub> T<sub>a</sub>ki<sub>ng</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>ons</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>per-roun</sub>d u<sub>pp</sub>er <sup>b</sup>oun<sup>d</sup>

$$
\mathbb { E } _ { \theta } \mathrm { K L } \Big ( \mathrm { B e r } ( \sigma ( \eta _ { t + 1 } ( \theta ) ) ) \Big \| \mathrm { B e r } ( p _ { \theta , t } ^ { \star , h } ) \Big ) \le \frac { 1 } { 8 } \sum _ { h < j \le t } \theta _ { j } ^ { 2 } .
$$

F<sub>or</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>,</sub> Pi<sub>ns</sub>k<sub>er</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> f<sub>or</sub> B<sub>ernou</sub>lli di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\begin{array} { r } { \mathbb { E } \Big [ \mathrm { K L } \Big ( \mathrm { B e r } ( \sigma ( R _ { t , h } + W _ { t , h } ) ) \Big | \Big | \mathrm { B e r } ( p _ { \theta , t } ^ { \star , h } ) \Big ) \Big | \mathcal { G } _ { t , h } \Big ] \geq 2 \mathrm { ~ V a r } \big ( \sigma ( R _ { t , h } + W _ { t , h } ) \mid \mathcal { G } _ { t , h } \big ) . } \end{array}
$$

L<sub>e</sub>t $W _ { t , h } ^ { \prime }$ b<sub>e</sub> <sub>an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>con</sub>diti<sub>ona</sub>l <sub>copy</sub> <sub>o</sub>f $W _ { t , h }$ <sub>.</sub> Si<sub>nce</sub> <sub>every</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f $R _ { t , h } + W _ { t , h }$ li<sub>es</sub> i<sub>n</sub> $[ - B , \dot { B } ]$ <sub>an</sub>d $\sigma ^ { \prime } ( z ) \geq \kappa _ { B }$ <sub>on</sub> thi<sub>s</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>,</sub> th<sub>e</sub> <sub>mean-va</sub>l<sub>ue</sub> th<sub>eorem</sub> <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r l } & { \mathrm { V a r } ( \sigma ( R _ { t , h } + W _ { t , h } ) \mid \mathcal { G } _ { t , h } ) = \cfrac { 1 } { 2 } \mathbb { E } \big [ ( \sigma ( R _ { t , h } + W _ { t , h } ) - \sigma ( R _ { t , h } + W _ { t , h } ^ { \prime } ) ) ^ { 2 } \big | \mathcal { G } _ { t , h } \big ] } \\ & { \qquad \geq \cfrac { \kappa _ { B } ^ { 2 } } { 2 } \mathbb { E } [ ( W _ { t , h } - W _ { t , h } ^ { \prime } ) ^ { 2 } \mid \mathcal { G } _ { t , h } ] } \\ & { \qquad = \kappa _ { B } ^ { 2 } \mathrm { V a r } ( W _ { t , h } \mid \mathcal { G } _ { t , h } ) . } \end{array}
$$

Th<sub>us</sub> th<sub>e per-roun</sub>d di<sub>s</sub>t<sub>or</sub>ti<sub>on</sub> i<sub>s a</sub>t l<sub>eas</sub>t $2 \kappa _ { B } ^ { 2 } \textstyle \sum _ { h < j \leq t } \theta _ { j } ^ { 2 }$ <sub>.</sub> S<sub>umm</sub>i<sub>ng</sub> th<sub>e</sub> <sub>upper</sub> <sub>an</sub>d l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> <sub>over</sub> $t = 1 , \dots , T$ and reversin<sub>g</sub> the order of summation <sub>g</sub>ives (28). □

## Appendix B Proof of Theorem 4.1

Th<sub>e proo</sub>f h<sub>as</sub> t<sub>wo</sub> i<sub>ngre</sub>di<sub>en</sub>t<sub>s.</sub> L<sub>emma</sub> B<sub>.</sub>1 fi<sub>rs</sub>t <sub>compares</sub> th<sub>e</sub> B<sub>ayes</sub>i<sub>an m</sub>i<sub>x</sub>t<sub>ure w</sub>ith <sub>any</sub> l<sub>oca</sub>li<sub>ze</sub>d di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>over</sub> <sub>parame</sub>t<sub>ers.</sub> W<sub>e</sub> th<sub>en</sub> <sub>c</sub>h<sub>oose</sub> th<sub>a</sub>t di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> i<sub>n</sub> <sub>eac</sub>h <sub>coor</sub>di<sub>na</sub>t<sub>e,</sub> <sub>a</sub>t <sub>a</sub> <sub>sca</sub>l<sub>e</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> $n _ { T , j } r _ { j } ^ { 2 } .$

## B.1 Auxiliary variational bound for Theorem 4.1

Lemma B.1 (Variational mixture bound). Let � and $( Q _ { \nu } ) _ { \nu \in \mathcal { V } }$ be probability measures dominated by a common measure, let � be a probability measure on $\mathcal { V } ,$ , and set $\begin{array} { r } { M = \int Q _ { \nu } \pi ( d \nu ) } \end{array}$ . Then,for every $\rho \ll \pi$

$$
\mathrm { K L } ( P \| M ) \leq \int \mathrm { K L } ( P \| Q _ { \nu } ) \rho ( d \nu ) + \mathrm { K L } ( \rho \| \pi ) .\tag{88}
$$

Proof. Write $p , q _ { \nu } , m$ f<sub>or</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> d<sub>ens</sub>iti<sub>es</sub> <sub>an</sub>d l<sub>e</sub>t $f = d \rho / d \pi$ <sub>.</sub> F<sub>o</sub>r �<sub>-a</sub>lm<sub>os</sub>t <sub>eve</sub>r<sub>y</sub> <sub>�,</sub>

$$
\begin{array} { l } { \displaystyle \log \frac { p ( x ) } { m ( x ) } \leq - \log \int _ { \{ f > 0 \} } \frac { q _ { \nu } ( x ) } { p ( x ) f ( \nu ) } \rho ( d \nu ) } \\ { \leq \displaystyle \int \log \biggl ( \frac { p ( x ) } { q _ { \nu } ( x ) } f ( \nu ) \biggr ) \rho ( d \nu ) , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> di<sub>scar</sub>d<sub>s</sub> th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f $\{ f = 0 \}$ f<sub>rom</sub> th<sub>e</sub> <sub>nonnega</sub>ti<sub>ve</sub> <sub>�-</sub>i<sub>n</sub>t<sub>egra</sub>l<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>secon</sub>d a<sub>pp</sub>lies Jensen’s ine<sub>q</sub>ualit<sub>y</sub> to the convex function − lo<sub>g</sub>. Inte<sub>g</sub>ratin<sub>g</sub> with res<sub>p</sub>ect to � and usin<sub>g</sub> Fubini <sub>p</sub>roves th<sub>e c</sub>l<sub>a</sub>i<sub>m.</sub> Th<sub>e usua</sub>l <sub>ex</sub>t<sub>en</sub>d<sub>e</sub>d<sub>-va</sub>l<sub>ue conven</sub>ti<sub>on covers po</sub>i<sub>n</sub>t<sub>s a</sub>t <sub>w</sub>hi<sub>c</sub>h <sub>a</sub> d<sub>ens</sub>it<sub>y van</sub>i<sub>s</sub>h<sub>es.</sub> □

## B.2 Proof of Theorem 4.1: Coordinate localization

ProofofTheorem 4.1. We proceed in four steps: define the statistical scale of each coordinate, construct a l<sub>oca</sub>li<sub>ze</sub>d <sub>pro</sub>d<sub>uc</sub>t di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on,</sub> b<sub>oun</sub>d it<sub>s</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>an</sub>d l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> <sub>cos</sub>t<sub>s,</sub> <sub>an</sub>d <sub>compare</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>sum</sub> <sub>w</sub>ith $\Gamma _ { T } ( r )$ .

Step 1: coordinate scales. Fix $\theta \in \Theta ( r )$ <sub>.</sub> A h<sub>or</sub>i<sub>zon-</sub>� <sub>source</sub> l<sub>aw</sub> d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> $\theta _ { 1 } , \ldots , \theta _ { T }$ <sub>, an</sub>d <sub>coor</sub>di<sub>na</sub>t<sub>es w</sub>ith $r _ { j } = 0$ <sub>are</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c.</sub> F<sub>or every rema</sub>i<sub>n</sub>i<sub>ng</sub> $1 \leq j \leq T$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
n _ { j } = n _ { T , j } , \qquad s _ { j } = n _ { j } r _ { j } ^ { 2 } .\tag{89}
$$

H<sub>ere</sub> $n _ { j }$ is exactly the number of prediction rounds on which lag � appears, while $s _ { j }$ i<sub>s</sub> it<sub>s</sub> <sub>square</sub>d <sub>parame</sub>t<sub>er</sub> <sub>ra</sub>di<sub>us measure</sub>d <sub>a</sub>t th<sub>e</sub> $n _ { i } ^ { - 1 / 2 }$ <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>reso</sub>l<sub>u</sub>ti<sub>on.</sub> L<sub>e</sub>t $\pi _ { j }$ b<sub>e un</sub>if<sub>orm on</sub> $[ - r _ { j } , r _ { j } ] , \mathrm { s o } \pi _ { T } = \bigotimes _ { j = 1 } ^ { T } \pi _ { j }$ i<sub>s</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> $Q _ { \pi _ { T } }$ i<sub>n</sub> $( 3 \dot { 1 } )$ )

Step 2: localized variational distribution. We construct a product variational distribution $\rho = \bigotimes _ { j = 1 } ^ { T } \rho _ { j }$ If $s _ { j } \leq 1$ , set $\rho _ { j } = \pi _ { j }$ <sub>.</sub> If $s _ { j } > 1$ <sub>,</sub> th<sub>en</sub> $\delta _ { j } = n _ { i } ^ { - 1 / 2 } < r _ { j }$ <sub>.</sub> Ch<sub>oose any</sub> i<sub>n</sub>t<sub>erva</sub>l $I _ { j } \subset [ - r _ { j } , r _ { j } ]$ <sub>o</sub>f l<sub>eng</sub>th $\delta _ { j }$ th<sub>a</sub>t <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> $\theta _ { j }$ <sub>an</sub>d l<sub>e</sub>t $\rho _ { j }$ b<sub>e un</sub>if<sub>orm on</sub> $I _ { j }$ <sub>.</sub> S<sub>uc</sub>h <sub>an</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>a</sub>l<sub>ways ex</sub>i<sub>s</sub>t<sub>s: use</sub> $[ \theta _ { j } , \theta _ { j } + \delta _ { j } ]$ <sub>w</sub>h<sub>en</sub> $\theta _ { j } \leq r _ { j } - \delta _ { j }$ <sub>an</sub>d $[ \theta _ { j } - \delta _ { j } , \theta _ { j } ]$ <sub>o</sub>th<sub>erw</sub>i<sub>se.</sub>

F<sub>or an ac</sub>ti<sub>ve coor</sub>di<sub>na</sub>t<sub>e</sub> $s _ { j } ~ > ~ 1$ , <sup>ever</sup>y $\nu _ { j }$ i<sub>n</sub> th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d i<sub>n</sub>t<sub>erva</sub>l li<sub>es w</sub>ithi<sub>n</sub> $n _ { j } ^ { - 1 / 2 }$ <sub>o</sub>f $\theta _ { j }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>pr</sub>i<sub>or-</sub>t<sub>o-pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>vo</sub>l<sub>ume</sub> <sub>ra</sub>ti<sub>o</sub> i<sub>s</sub> $2 r _ { j } \sqrt { n _ { j } }$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
\mathbb { E } _ { \rho } { \left( \nu _ { j } - \theta _ { j } \right) } ^ { 2 } \le \delta _ { j } ^ { 2 } = \frac { 1 } { n _ { j } } , \qquad \mathrm { K L } ( \rho _ { j } \| \pi _ { j } ) = \log \frac { 2 r _ { j } } { \delta _ { j } } = \frac { 1 } { 2 } \log s _ { j } + \log 2 .\tag{90}
$$

F<sub>or an</sub> i<sub>nac</sub>ti<sub>ve coor</sub>di<sub>na</sub>t<sub>e</sub> $s _ { j } \leq 1$ <sub>,</sub> <sub>no</sub> l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> i<sub>s</sub> <sub>nee</sub>d<sub>e</sub>d<sub>.</sub> Si<sub>nce</sub> $\nu _ { j }$ i<sub>s</sub> <sub>un</sub>if<sub>orm</sub> <sub>on</sub> $[ - r _ { j } , r _ { j } ]$

$$
\mathbb { E } _ { \rho } ( \nu _ { j } - \theta _ { j } ) ^ { 2 } = \frac { r _ { j } ^ { 2 } } { 3 } + \theta _ { j } ^ { 2 } \le \frac { 4 } { 3 } r _ { j } ^ { 2 } \le 4 r _ { j } ^ { 2 } , \qquad \mathrm { K L } ( \rho _ { j } \| \pi _ { j } ) = 0 .\tag{91}
$$

Step 3: approximation and localization costs. The global upper half of Lemma 3.2 turns squared <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>error</sub> i<sub>n</sub>t<sub>o</sub> <sub>sequence-</sub>l<sub>eve</sub>l KL di<sub>vergence.</sub> I<sub>n</sub>d<sub>epen</sub>d<sub>ence</sub> <sub>un</sub>d<sub>er</sub> $\rho$ th<sub>en</sub> <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { l } { \displaystyle { \int \mathrm { K L } ( P _ { \theta , T } | | P _ { \nu , T } ) \rho ( d \nu ) \le \frac { 1 } { 8 } \sum _ { j = 1 } ^ { T } n _ { j } \mathbb { E } _ { \rho } ( \nu _ { j } - \theta _ { j } ) ^ { 2 } } } \\ { \displaystyle { \qquad \le \frac { 1 } { 8 } | \{ j : s _ { j } > 1 \} | + \frac { 1 } { 2 } \sum _ { j : s _ { j } \le 1 } s _ { j } } . } \end{array}\tag{92}
$$

M<sub>oreover,</sub>

$$
\mathrm { K L } ( \rho \| \pi _ { T } ) = \sum _ { j : s _ { j } > 1 } \left( \frac { 1 } { 2 } \log s _ { j } + \log 2 \right) .\tag{93}
$$

A<sub>pp</sub>l<sub>y</sub> L<sub>emma</sub> B<sub>.</sub>1 <sub>w</sub>ith $P = P _ { \theta , T }$ <sub>an</sub>d $Q _ { \nu } = P _ { \nu , T }$ . Combinin<sub>g</sub> (92) and (93) <sub>g</sub>ives

$$
\begin{array} { r l r } {  { \mathrm { K L } ( P _ { \theta , T } \| Q _ { \pi _ { T } } ) \le \displaystyle \frac { 1 } { 2 } \sum _ { j : s _ { j } > 1 } \log s _ { j } + C ( | \{ j : s _ { j } > 1 \} | + \sum _ { j : s _ { j } \le 1 } s _ { j } ) } } \\ & { } & { \le \displaystyle \frac { 1 } { 2 } \sum _ { j = 1 } ^ { T } \log ( 1 + s _ { j } ) + C d _ { T } ( r ) . } \end{array}\tag{94}
$$

Step 4: comparison with the spectrum. For $s _ { j } > 1$ <sub>,</sub> th<sub>e</sub> l<sub>ea</sub>di<sub>ng</sub> l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> <sub>cos</sub>t i<sub>s</sub> $\frac { 1 } { 2 }$ <sup>l</sup>o<sub>g</sub> $s _ { j } ; $ f<sub>or</sub> $s _ { j } \leq 1$ <sub>,</sub> th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on cos</sub>t i<sub>s</sub> $O ( s _ { j } )$ <sub>.</sub> Th<sub>ese</sub> t<sub>wo cases are s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y represen</sub>t<sub>e</sub>d b<sub>y</sub> $\log ( 1 + s _ { j } )$ <sub>.</sub> M<sub>ore</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y,</sub>

$$
\operatorname* { m i n } \{ 1 , s \} \leq \left\{ { 2 \log ( 1 + s ) } , \quad \begin{array} { l l } { 0 \leq s \leq 1 , } \\ { ( \log 2 ) ^ { - 1 } \log ( 1 + s ) , } & { s > 1 , } \end{array}  \right.\tag{95}
$$

so $d _ { T } ( r ) \leq C \Gamma _ { T } ( r )$ . Equation (94) <sub>p</sub>roves the first inequalit<sub>y</sub> in (32); the dis<sub>p</sub>la<sub>y</sub>ed com<sub>p</sub>arison <sub>p</sub>roves its <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y.</sub> Th<sub>e</sub> b<sub>oun</sub>d i<sub>s</sub> <sub>un</sub>if<sub>orm</sub> <sub>over</sub> $\theta \in \Theta ( r )$ <sub>,</sub> <sub>so</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> <sub>supremum</sub> <sub>an</sub>d th<sub>en</sub> th<sub>e</sub> i<sub>n</sub>fi<sub>mum</sub> <sub>over</sub> all <sub>p</sub>redictors <sub>p</sub>roves (33). □

## Appendix C Proof of Lemma 4.4

W<sub>e</sub> <sub>prove</sub> L<sub>emma</sub> 4<sub>.</sub>4<sub>.</sub> Th<sub>e</sub> <sub>on</sub>l<sub>y</sub> <sub>pro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> i<sub>ngre</sub>di<sub>en</sub>t i<sub>s</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>e</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on.</sub>

## C.1 Auxiliary forest-product lemma for Lemma 4.4

Lemma C.1 (Edge products on a forest). Let $G = ( V , E )$ be a finite forest and let $( U _ { \nu } ) _ { \nu \in V }$ be independent Rademacher random variables. Then the edge products

$$
W _ { \{ \nu , w \} } = U _ { \nu } U _ { w } , \qquad \{ \nu , w \} \in E ,\tag{96}
$$

are independent Rademacher random variables.

Proof. Choose one root in each connected component. The map

$$
( U _ { \nu } ) _ { \nu \in V } \longmapsto \left( ( U _ { \mathrm { r o o t } ( C ) } ) _ { C } , ( U _ { \nu } U _ { w } ) _ { \{ \nu , w \} \in E } \right)\tag{97}
$$

from vertex signs to root signs and edge products is bijective. Indeed, starting from a root sign, the sign at <sub>every</sub> <sub>o</sub>th<sub>er</sub> <sub>ver</sub>t<sub>ex</sub> i<sub>s</sub> <sub>recovere</sub>d <sub>un</sub>i<sub>que</sub>l<sub>y</sub> b<sub>y</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> <sub>e</sub>d<sub>ge</sub> <sub>pro</sub>d<sub>uc</sub>t<sub>s</sub> <sub>a</sub>l<sub>ong</sub> th<sub>e</sub> <sub>un</sub>i<sub>que</sub> <sub>roo</sub>t<sub>-</sub>t<sub>o-ver</sub>t<sub>ex</sub> <sub>pa</sub>th<sub>.</sub> Si<sub>nce</sub> th<sub>e</sub> i<sub>npu</sub>t <sub>vec</sub>t<sub>or</sub> i<sub>s un</sub>if<sub>orm on</sub> $\{ - 1 , + 1 \} ^ { | V | }$ , the output of the bijection is also uniform on a hypercube. I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar, a</sub>ll <sub>e</sub>d<sub>ge-pro</sub>d<sub>uc</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>es are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t R<sub>a</sub>d<sub>emac</sub>h<sub>ers.</sub> □

## C.2 Proof of Lemma 4.4: Toeplitz Gram concentration

Th<sub>e</sub> di<sub>agona</sub>l <sub>en</sub>t<sub>r</sub>i<sub>es o</sub>f th<sub>e</sub> G<sub>ram ma</sub>t<sub>r</sub>i<sub>x are</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c:</sub>

$$
\big ( ( Z ^ { ( J ) } ) ^ { \top } Z ^ { ( J ) } \big ) _ { j , j } = \sum _ { t = 1 } ^ { N } U _ { J + t - j } ^ { 2 } = N .\tag{98}
$$

Fi<sub>x</sub> $j \neq k$ an<sup>d</sup> <sub>p</sub>ut $d = | j - k | \geq 1$ <sub>.</sub> Aft<sub>er</sub> <sub>re</sub>l<sub>a</sub>b<sub>e</sub>li<sub>ng</sub> th<sub>e</sub> <sub>s</sub>t<sub>ar</sub>ti<sub>ng</sub> i<sub>n</sub>d<sub>ex,</sub> th<sub>e</sub> <sub>o</sub>f<sub>-</sub>di<sub>agona</sub>l <sub>en</sub>t<sub>ry</sub> h<sub>as</sub> th<sub>e</sub> f<sub>orm</sub>

$$
S _ { j , k } = \sum _ { s = a } ^ { a + N - 1 } U _ { s } U _ { s + d }\tag{99}
$$

for an inte<sub>g</sub>er �. Consider the <sub>g</sub>ra<sub>p</sub>h whose vertices are the in<sub>p</sub>ut indices a<sub>pp</sub>earin<sub>g</sub> in (99) and whose ed<sub>g</sub>es are $\{ s , s + d \} , s = a , \ldots , a + N - 1$ <sub>.</sub> Ed<sub>ges preserve res</sub>id<sub>ue c</sub>l<sub>asses mo</sub>d<sub>u</sub>l<sub>o</sub> $d ,$ <sub>an</sub>d <sub>w</sub>ithi<sub>n eac</sub>h <sub>res</sub>id<sub>ue c</sub>l<sub>ass</sub> they connect points in increasing order along the integer line. The graph is therefore a disjoint union of paths and in <sub>p</sub>articular a forest. B<sub>y</sub> Lemma C.1, the � summands in (99) are inde<sub>p</sub>endent Rademacher variables.

H<sub>oe</sub>fdi<sub>ng</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>consequen</sub>tl<sub>y</sub> <sub>g</sub>i<sub>ves</sub>

$$
\mathbb { P } \left( | S _ { j , k } | > \frac { N } { 2 J } \right) \le 2 \exp \left( - \frac { N } { 8 J ^ { 2 } } \right) .\tag{100}
$$

Th<sub>ere are</sub> f<sub>ewer</sub> th<sub>an</sub> $J ^ { 2 }$ <sub>or</sub>d<sub>ere</sub>d <sub>o</sub>f<sub>-</sub>di<sub>agona</sub>l <sub>pa</sub>i<sub>rs.</sub> A <sub>un</sub>i<sub>on</sub> b<sub>oun</sub>d <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t<sub>, w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t th<sub>e</sub> ri<sub>g</sub>ht side of (44), ever<sub>y</sub> of-dia<sub>g</sub>onal entr<sub>y</sub> has ma<sub>g</sub>nitude at most $N / ( 2 J )$ <sub>.</sub> O<sub>n</sub> thi<sub>s even</sub>t th<sub>e sum o</sub>f th<sub>e</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e o</sub>f<sub>-</sub>di<sub>agona</sub>l <sub>en</sub>t<sub>r</sub>i<sub>es</sub> i<sub>n any row</sub> i<sub>s a</sub>t <sub>mos</sub>t

$$
( J - 1 ) \frac { N } { 2 J } < \frac { N } { 2 } .\tag{101}
$$

Gersh<sub>g</sub>orin’s circle theorem and (98) im<sub>p</sub>l<sub>y</sub> that ever<sub>y</sub> ei<sub>g</sub>envalue lies in $\left[ N / 2 , 3 N / 2 \right]$ . This is exactl<sub>y</sub> (44). F<sub>or</sub> $J = 1$ <sub>,</sub> th<sub>e</sub> <sub>conc</sub>l<sub>us</sub>i<sub>on</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c.</sub>

## Appendix D Proofs of Theorems 4.3 and 4.5

Thi<sub>s appen</sub>di<sub>x proves</sub> th<sub>e genera</sub>l <sub>exogenous-</sub>d<sub>es</sub>i<sub>gn converse an</sub>d th<sub>en spec</sub>i<sub>a</sub>li<sub>zes</sub> it t<sub>o</sub> th<sub>e</sub> l<sub>agge</sub>d <sub>source.</sub>

## D.1 Auxiliary information inequalities for Theorems 4.3 and 4.5

Lemma D.1 (Bayesian information lower bound). Let $( P _ { \vartheta } ) _ { \vartheta \in \mathcal { V } }$ be any dominated family and let Π be a prior on V. $I f \Theta \sim \Pi$ and $D \mid \Theta = \vartheta \sim P _ { \vartheta }$ , then

$$
\operatorname* { i n f } _ { Q } \operatorname* { s u p } _ { \vartheta \in \mathcal { V } } \mathrm { K L } ( P _ { \vartheta } \| Q ) \geq I ( \Theta ; D ) .\tag{102}
$$

Proof. Let $\begin{array} { r } { P _ { \Pi } = \int P _ { \vartheta } \Pi ( d \vartheta ) } \end{array}$ b<sub>e</sub> th<sub>e</sub> <sub>m</sub>i<sub>x</sub>t<sub>ure</sub> l<sub>aw.</sub> F<sub>or</sub> <sub>every</sub> $Q$

$$
\begin{array} { r l } {  { \int \operatorname { K L } ( P _ { \vartheta } \| Q ) \Pi ( d \vartheta ) = \int \operatorname { K L } ( P _ { \vartheta } \| P _ { \Pi } ) \Pi ( d \vartheta ) + \operatorname { K L } ( P _ { \Pi } \| Q ) } } \\ & { = I ( \Theta ; D ) + \operatorname { K L } ( P _ { \Pi } \| Q ) } \\ & { \geq I ( \Theta ; D ) . } \end{array}
$$

The supremum is at least the prior average. Taking the infimum over � proves the claim.

Lemma D.2 (Entropy–MSE bound). Let Θ be uniform on the rectangle $\Pi _ { j = 1 } ^ { J } [ - r _ { j } / 2 , r _ { j } / 2 ]$ , with every $r _ { j } > 0$ and let $\widehat { \Theta }$ be any estimator constructed from data �. Then

$$
I ( \Theta ; D ) \geq \sum _ { j = 1 } ^ { J } \log r _ { j } - \frac { J } { 2 } \log \left( 2 \pi e ^ { \frac { \mathbb { E } \| \Theta - \widehat { \Theta } \| _ { 2 } ^ { 2 } } { J } } \right) .\tag{103}
$$

The inequality is understood in the extended sense when an error distribution is singular.

Proof. Data processing gives $I ( \Theta ; D ) \ge I ( \Theta ; \widehat { \Theta } )$ <sub>.</sub> Si<sub>nce</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> h<sub>as</sub> dif<sub>eren</sub>ti<sub>a</sub>l <sub>en</sub>t<sub>ropy</sub> $h ( \Theta ) = \textstyle \sum _ { j }$ <sup>l</sup>o<sub>g</sub> $r _ { j }$

$$
\begin{array} { r l } & { I ( \Theta ; \widehat { \Theta } ) = h ( \Theta ) - h ( \Theta \mid \widehat { \Theta } ) } \\ & { \qquad = h ( \Theta ) - h ( \Theta - \widehat { \Theta } \mid \widehat { \Theta } ) } \\ & { \qquad \geq h ( \Theta ) - h ( \Theta - \widehat { \Theta } ) . } \end{array}
$$

F<sub>or any</sub> �<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>ran</sub>d<sub>om vec</sub>t<sub>or</sub> �<sub>,</sub> G<sub>auss</sub>i<sub>an max</sub>i<sub>ma</sub>l <sub>en</sub>t<sub>ropy,</sub> th<sub>e</sub> b<sub>oun</sub>d t<sub>r</sub> C<sub>ov</sub> $( W ) \leq \mathbb { E } \| W \| _ { 2 } ^ { 2 }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>ar</sub>ith<sub>me</sub>ti<sub>c–geome</sub>t<sub>r</sub>i<sub>c</sub> <sub>mean</sub> i<sub>nequa</sub>lit<sub>y</sub> f<sub>or</sub> <sub>covar</sub>i<sub>ance</sub> <sub>e</sub>i<sub>genva</sub>l<sub>ues</sub> <sub>g</sub>i<sub>ve</sub>

$$
h ( W ) \leq \frac { J } { 2 } \log \left( 2 \pi e \frac { \mathbb { E } \| W \| _ { 2 } ^ { 2 } } { J } \right) .\tag{104}
$$

A<sub>pp</sub>l<sub>y</sub> thi<sub>s</sub> <sub>w</sub>ith $W = \Theta - \widehat { \Theta }$

## D.2 Proof of Theorem 4.3: Conditioned-design information bound

Proof of Theorem 4.3. For $\nu \in K _ { J }$ <sub>,</sub> l<sub>e</sub>t

$$
\ell _ { N } ( \nu ) = \sum _ { i = 1 } ^ { N } \left[ \psi ( b _ { i } + z _ { i } ^ { \top } \nu ) - Y _ { i } ( b _ { i } + z _ { i } ^ { \top } \nu ) \right]\tag{105}
$$

b<sub>e</sub> th<sub>e con</sub>diti<sub>ona</sub>l <sub>nega</sub>ti<sub>ve</sub> l<sub>og</sub> lik<sub>e</sub>lih<sub>oo</sub>d<sub>.</sub> Ch<sub>oose a measura</sub>bl<sub>e m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer</sub>

$$
\widehat { \Theta } \in \mathop { \mathrm { a r g } } _ { \nu \in K _ { J } } \ell _ { N } ( \nu ) ,
$$

b<sub>rea</sub>ki<sub>ng</sub> ti<sub>es</sub> l<sub>ex</sub>i<sub>cograp</sub>hi<sub>ca</sub>ll<sub>y.</sub> L<sub>e</sub>t

$$
\mathcal { E } = \{ \lambda _ { \operatorname* { m i n } } ( Z ^ { \top } Z ) \geq \mu N \} .\tag{106}
$$

B<sub>y</sub> th<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>-</sub>l<sub>og</sub>it <sub>assump</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> $\begin{array} { r } { \kappa _ { B _ { 0 } } = \operatorname* { m i n } _ { | x | \leq B _ { 0 } } \psi ^ { \prime \prime } ( x ) > 0 , } \end{array}$

$$
\nabla ^ { 2 } \ell _ { N } \left( \boldsymbol { \nu } \right) = \sum _ { i = 1 } ^ { N } \psi ^ { \prime \prime } ( b _ { i } + z _ { i } ^ { \top } \boldsymbol { \nu } ) z _ { i } z _ { i } ^ { \top } \succeq \kappa _ { B _ { 0 } } Z ^ { \top } Z .\tag{107}
$$

H<sub>ence on</sub> $\mathcal { E } , \ell _ { N }$ <sup>i</sup>s �-stron<sub>g</sub><sup>l</sup><sub>y</sub> convex over $K _ { J }$ <sub>w</sub>ith $m = \kappa _ { B _ { 0 } } \mu N$

W<sub>r</sub>it<sub>e</sub> $g = \nabla \ell _ { N } ( \Theta )$ . Since both Θ and $\widehat { \Theta }$ li<sub>e</sub> i<sub>n</sub> $K _ { J }$ <sub>an</sub>d $\ell _ { N } ( \widehat { \Theta } ) \leq \ell _ { N } ( \Theta )$ , <sup>stron</sup>g <sup>con</sup>v<sup>exit</sup>y g<sup>i</sup>v<sup>es</sup>

$$
0 \geq g ^ { \top } ( \widehat { \Theta } - \Theta ) + \frac { m } { 2 } \| \widehat { \Theta } - \Theta \| _ { 2 } ^ { 2 } .\tag{108}
$$

Th<sub>ere</sub>f<sub>ore,</sub> <sub>on</sub> $\mathcal { E } .$

$$
\lVert \widehat { \Theta } - \Theta \rVert _ { 2 } \leq \frac { 2 \lVert g \rVert _ { 2 } } { \kappa _ { B _ { 0 } } \mu N } .\tag{109}
$$

Conditional on (�, �, Θ),

$$
g = \sum _ { i = 1 } ^ { N } \left( \sigma ( b _ { i } + \boldsymbol { z } _ { i } ^ { \top } \Theta ) - Y _ { i } \right) \boldsymbol { z } _ { i }\tag{110}
$$

<sub>w</sub>ith i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>cen</sub>t<sub>ere</sub>d <sub>summan</sub>d<sub>s.</sub> Th<sub>us</sub>

$$
\mathbb { E } [ \| g \| _ { 2 } ^ { 2 } \mid Z , b , \Theta ] = \sum _ { i = 1 } ^ { N } \operatorname { V a r } ( Y _ { i } \mid Z , b , \Theta ) \| z _ { i } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { 4 } \operatorname { t r } ( Z ^ { \top } Z ) .\tag{111}
$$

Combinin<sub>g</sub> (109), (111), and (39) <sub>y</sub>ields

$$
\mathbb { E } \bigg [ \| \widehat { \Theta } - \Theta \| _ { 2 } ^ { 2 } { \mathbf { 1 } } _ { \mathcal { E } } \bigg ] \leq \frac { \tau } { \kappa _ { B _ { 0 } } ^ { 2 } \mu ^ { 2 } } \frac { J } { N } .\tag{112}
$$

On $\varepsilon ^ { c }$ <sub>,</sub> th<sub>e</sub> <sub>square</sub>d di<sub>ame</sub>t<sub>er</sub> <sub>o</sub>f $K _ { J }$ i<sub>s</sub> $\Delta _ { J } ^ { 2 }$ , so

$$
\mathbb { E } \bigg [ \| \widehat { \Theta } - \Theta \| _ { 2 } ^ { 2 } \mathbf { 1 } _ { \mathcal { E } ^ { c } } \bigg ] \leq \Delta _ { J } ^ { 2 } \delta \leq \frac { \tau J } { N } .\tag{113}
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
\mathbb { E } \Vert \widehat { \Theta } - \Theta \Vert _ { 2 } ^ { 2 } \leq C _ { B _ { 0 } , \mu , \tau } \frac { J } { N } .\tag{114}
$$

A<sub>pp</sub>l<sub>y</sub> Lemma D.2. Equation (114) <sub>g</sub>ives

$$
\begin{array} { l } { { I ( \Theta ; D ) \ge \displaystyle \sum _ { j = 1 } ^ { J } \log r _ { j } + \frac { J } { 2 } \log N - C _ { B _ { 0 } , \mu , \tau } J } } \\ { { { } } } \\ { { { } = \displaystyle \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } \log ( N r _ { j } ^ { 2 } ) - C _ { B _ { 0 } , \mu , \tau } J . } } \end{array}\tag{115}
$$

Mutual information is nonne<sub>g</sub>ative, which <sub>p</sub>ermits the <sub>p</sub>ositive <sub>p</sub>art in (42). Finall<sub>y</sub>, Lemma D.1 transfers the <sub>same</sub> b<sub>oun</sub>d t<sub>o</sub> $\Re _ { N } ( K _ { J } )$ . □

## D.3 Proof of Theorem 4.5: Spectrum lower bound for the lagged source

ProofofTheorem 4.5. Because � is nonincreasing and $r _ { J } > 0$ <sub>, we</sub> h<sub>ave</sub> $r _ { j } > 0$ <sup>f</sup>or ever<sub>y</sub> $1 \leq j \leq J$ <sub>.</sub> R<sub>es</sub>t<sub>r</sub>i<sub>c</sub>t t<sup>h</sup>e source <sub>p</sub>arameter to

$$
K _ { J } = \prod _ { j = 1 } ^ { J } [ - r _ { j } / 2 , r _ { j } / 2 ]\tag{116}
$$

and set ever<sub>y</sub> later coeficient to zero. Let Θ be uniform on $K _ { J }$ <sub>.</sub> R<sub>e</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub> i<sub>npu</sub>t <sub>sequence</sub> <sub>an</sub>d th<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> f<sub>rom</sub> <sub>roun</sub>d<sub>s</sub> $t = J , \ldots , T$ <sub>.</sub> With

$$
z _ { i } = ( U _ { J + i - 1 } , U _ { J + i - 2 } , \ldots , U _ { i } ) ^ { \top } , \qquad i = 1 , \ldots , N ,\tag{117}
$$

th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>con</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> i<sub>npu</sub>t<sub>s,</sub>

$$
Y _ { J + i } \mid U _ { 1 : T } , \Theta \sim \mathrm { B e r } ( \sigma ( z _ { i } ^ { \top } \Theta ) ) , \qquad i = 1 , \ldots , N .\tag{118}
$$

Th<sub>e</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>w</sub>ith <sub>rows</sub> $z _ { i } ^ { \top }$ i<sub>s</sub> $Z ^ { ( J ) }$ in (43). It obe<sub>y</sub>s

$$
\mathrm { t r } ( ( Z ^ { ( J ) } ) ^ { \top } Z ^ { ( J ) } ) = N J\tag{119}
$$

d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>ca</sub>ll<sub>y,</sub> <sub>an</sub>d <sub>every</sub> $\nu \in K _ { J }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
| z _ { i } ^ { \top } \nu | \leq \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } r _ { j } \leq \frac { B } { 2 } .\tag{120}
$$

B<sub>y</sub> L<sub>emma</sub> 4<sub>.</sub>4<sub>,</sub> th<sub>e</sub> <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> <sub>assump</sub>ti<sub>on</sub> i<sub>n</sub> Th<sub>eorem</sub> 4<sub>.</sub>3 h<sub>o</sub>ld<sub>s</sub> <sub>w</sub>ith $\mu = 1 / 2$ <sub>an</sub>d

$$
\delta = 2 J ^ { 2 } \exp \left( - \frac { N } { 8 J ^ { 2 } } \right) .\tag{121}
$$

Al<sub>so,</sub> $\begin{array} { r } { \Delta _ { J } ^ { 2 } = \sum _ { j \leq J } r _ { j } ^ { 2 } \leq B ^ { 2 } } \end{array}$ . Choosin<sub>g</sub> the constant in (45) suficientl<sub>y</sub> lar<sub>g</sub>e as a function of � <sub>g</sub>ives

$$
B ^ { 2 } \delta \leq \frac { J } { N } .\tag{122}
$$

Th<sub>us</sub> Th<sub>eorem</sub> 4<sub>.</sub>3 i<sub>mp</sub>li<sub>es</sub>

$$
I ( \Theta ; D _ { J } ) \ge \left[ \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } \log ( N r _ { j } ^ { 2 } ) - C _ { B } J \right] _ { + } ,\tag{123}
$$

<sub>w</sub>h<sub>ere</sub> $D _ { J }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d i<sub>npu</sub>t<sub>s</sub> <sub>an</sub>d l<sub>a</sub>b<sub>e</sub>l<sub>s.</sub>

Th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d d<sub>a</sub>t<sub>a</sub> $D _ { J }$ <sub>are a measura</sub>bl<sub>e</sub> f<sub>unc</sub>ti<sub>on o</sub>f th<sub>e</sub> f<sub>u</sub>ll <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> $D \ = \ X _ { 1 : T + 1 }$ . Th<sub>ere</sub>f<sub>ore</sub> $I ( \Theta ; D ) \ge I ( \Theta ; D _ { J } )$ <sub>.</sub> Th<sub>e pr</sub>i<sub>or</sub> i<sub>s suppor</sub>t<sub>e</sub>d <sub>on a su</sub>b<sub>c</sub>l<sub>ass o</sub>f $\Theta ( r )$ <sub>, so</sub> L<sub>emma</sub> D<sub>.</sub>1 <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>source</sub> f<sub>am</sub>il<sub>y</sub> <sub>g</sub>i<sub>ves</sub>

$$
\mathcal { R } _ { T } ( r ) \geq I ( \Theta ; D ) \geq I ( \Theta ; D _ { J } ) ,\tag{124}
$$

which <sub>p</sub>roves (46).

## Appendix E Proofs of Corollaries 4.7–4.9 and Memory-Profile Sum Bounds

W<sub>e prove</sub> th<sub>e ra</sub>t<sub>e coro</sub>ll<sub>ar</sub>i<sub>es an</sub>d <sub>recor</sub>d th<sub>e correspon</sub>di<sub>ng memory-pro</sub>fil<sub>e sums.</sub>

## E.1 Proof of Corollary 4.7: Finite memory

If $r _ { j } = 0$ f<sub>or</sub> $j > k$ <sub>,</sub> Th<sub>eorem</sub> 4<sub>.</sub>1 <sub>an</sub>d $n _ { T , j } \leq T$ <sub>g</sub><sup>i</sup>ve

$$
\mathcal { R } _ { T } ( r ) \leq C \sum _ { j = 1 } ^ { k } \log ( 1 + n _ { T , j } r _ { j } ^ { 2 } ) \leq C \sum _ { j = 1 } ^ { k } \log ( 1 + T r _ { j } ^ { 2 } ) .\tag{125}
$$

Under (50), all � coordinates belon<sub>g</sub> to the stron<sub>g</sub>l<sub>y</sub> active set in Corollar<sub>y</sub> 4.6; the condition there follows <sub>a</sub>ft<sub>er</sub> <sub>c</sub>h<sub>ang</sub>i<sub>ng</sub> $C _ { B }$ . This <sub>g</sub>ives the lower half of (51). For fixed <sub>p</sub>ositive radii, ever<sub>y</sub> summand is lo<sub>g</sub> $T + O ( 1 )$ <sub>prov</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>asser</sub>ti<sub>on o</sub>f C<sub>oro</sub>ll<sub>ary</sub> 4<sub>.</sub>7<sub>.</sub>

## E.2 Proof of Corollary 4.8: Exponential envelope

P<sub>u</sub>t $\Lambda = \log ( A ^ { 2 } T )$ <sub>.</sub> F<sub>or</sub> th<sub>e</sub> <sub>upper</sub> b<sub>oun</sub>d<sub>,</sub> <sub>use</sub> $n _ { T , j } \leq T$ <sub>an</sub>d <sub>mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y</sub> t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { l } { \displaystyle \Gamma _ { T } ( r ) \leq \sum _ { j = 1 } ^ { \infty } \log ( 1 + A ^ { 2 } T e ^ { - 2 \alpha j } ) } \\ { \displaystyle \qquad \leq \int _ { 0 } ^ { \infty } \log ( 1 + A ^ { 2 } T e ^ { - 2 \alpha x } ) d x } \\ { \displaystyle \qquad = \frac 1 { 2 \alpha } \int _ { 0 } ^ { \infty } \log ( 1 + e ^ { \Lambda - y } ) d y . } \end{array}\tag{126}
$$

F<sub>or</sub> $0 \le y \le \Lambda , \log ( 1 + e ^ { \Lambda - y } ) \le \Lambda - y +$ + lo<sub>g</sub> 2; for $y > \Lambda$ <sub>,</sub> it i<sub>s a</sub>t <sub>mos</sub>t $e ^ { \Lambda - y }$ <sub>.</sub> H<sub>ence</sub>

$$
\Gamma _ { T } ( r ) \leq { \frac { 1 } { 2 \alpha } } \left( { \frac { \Lambda ^ { 2 } } { 2 } } + \Lambda \log 2 + 1 \right) \leq C { \frac { \Lambda ^ { 2 } } { \alpha } }\tag{127}
$$

<sub>w</sub>h<sub>enever</sub> $\Lambda \geq 1$ . The u<sub>pp</sub>er half of (54) follows from Theorem 4.1.

F<sub>or</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>,</sub> l<sub>e</sub>t

$$
J = \left\lfloor { \frac { \Lambda } { 8 \alpha } } \right\rfloor .\tag{128}
$$

The first condition in (53), with a suficientl<sub>y</sub> lar<sub>g</sub>e constant $C _ { B } .$ , ensures $J \geq 1$ <sub>an</sub>d $\Lambda \geq 4 ( C _ { B } + \log 2 + 1 )$ Aft<sub>er</sub> d<sub>ecreas</sub>i<sub>ng</sub> $c _ { B }$ if <sub>necessary,</sub> th<sub>e</sub> <sub>secon</sub>d <sub>con</sub>diti<sub>on</sub> <sub>ensures</sub> $J \le T / 2$ <sub>an</sub>d i<sub>mp</sub>li<sub>es</sub> th<sub>e</sub> di<sub>mens</sub>i<sub>on con</sub>diti<sub>on</sub> i<sub>n</sub> Th<sub>eorem</sub> 4<sub>.</sub>5<sub>.</sub> Si<sub>nce</sub> $N = T - J + 1 \geq T / 2$

$$
\begin{array} { c } { { \displaystyle \mathcal { R } _ { T } ( r ) \geq \left[ \frac { 1 } { 2 } \sum _ { j = 1 } ^ { J } \bigl ( \Lambda - \log 2 - 2 \alpha j \bigr ) - C _ { B } J \right] _ { + } } } \\ { { = \left[ \frac { J } { 2 } ( \Lambda - \log 2 ) - \frac { \alpha J ( J + 1 ) } { 2 } - C _ { B } J \right] _ { + } . } } \end{array}\tag{129}
$$

B<sub>y</sub> (128), the first term is of order $\Lambda ^ { 2 } / \alpha$ <sub>,</sub> th<sub>e qua</sub>d<sub>ra</sub>ti<sub>c</sub> t<sub>erm</sub> i<sub>s a</sub>t <sub>mos</sub>t <sub>a su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y sma</sub>ll <sub>cons</sub>t<sub>an</sub>t ti<sub>mes</sub> $\Lambda ^ { 2 } / \alpha$ , and the final two linear terms are absorbed because Λ is lar<sub>g</sub>er than a constant de<sub>p</sub>endin<sub>g</sub> on �. Thus

$$
\mathcal { R } _ { T } ( r ) \geq c _ { B } \frac { \Lambda ^ { 2 } } { \alpha } .\tag{130}
$$

Thi<sub>s</sub> <sub>proves</sub> C<sub>oro</sub>ll<sub>ary</sub> 4<sub>.</sub>8<sub>.</sub> F<sub>or</sub> fi<sub>xe</sub>d � <sub>an</sub>d $\alpha , \Lambda = \log T + O ( 1 )$ <sub>an</sub>d th<sub>e</sub> di<sub>mens</sub>i<sub>on con</sub>diti<sub>on</sub> h<sub>o</sub>ld<sub>s</sub> b<sub>ecause</sub> (log $T ) ^ { 3 } = o ( T )$ .

## E.3 Proof of Corollary 4.9: Polynomial envelope

L<sub>e</sub>t $M = ( A ^ { 2 } T ) ^ { 1 / ( 2 s ) }$ <sub>.</sub> A<sub>s</sub> b<sub>e</sub>f<sub>ore,</sub>

$$
\Gamma _ { T } ( r ) \leq \sum _ { j = 1 } ^ { \infty } \log \Bigl ( 1 + ( M / j ) ^ { 2 s } \Bigr ) .\tag{131}
$$

The first condition in (57) ma<sub>y</sub> be taken to ensure $M \geq 2$ <sub>.</sub> P<sub>u</sub>t $m = \lceil M \rceil$ <sub>.</sub> F<sub>or</sub> $j \leq m$

$$
\log \left( 1 + ( M / j ) ^ { 2 s } \right) \leq \log 2 + 2 s \log ( m / j ) .\tag{132}
$$

Usin<sub>g</sub> $\log ( m ! ) \geq m$ <sup>l</sup>o<sub>g</sub> $m - m$

$$
\begin{array} { l } { \displaystyle \sum _ { j = 1 } ^ { m } \log \left( 1 + ( M / j ) ^ { 2 s } \right) \leq m \log 2 + 2 s \big ( m \log m - \log ( m ! ) \big ) } \\ { \leq C _ { s } m \leq C _ { s } M . } \end{array}\tag{133}
$$

F<sub>or</sub> th<sub>e</sub> t<sub>a</sub>il<sub>,</sub> l<sub>og</sub> $( 1 + x ) \leq x$ <sub>an</sub>d th<sub>e</sub> i<sub>n</sub>t<sub>egra</sub>l t<sub>es</sub>t <sub>g</sub>i<sub>ve</sub>

$$
\begin{array} { r } { \displaystyle \sum _ { j > m } \log \Bigl ( 1 + ( M / j ) ^ { 2 s } \Bigr ) \leq M ^ { 2 s } \sum _ { j > m } j ^ { - 2 s } \qquad } \\ { \leq C _ { s } M ^ { 2 s } m ^ { 1 - 2 s } \leq C _ { s } M . } \end{array}\tag{134}
$$

Th<sub>us</sub> $\Gamma _ { T } ( r ) \leq C _ { s } M$ , and Theorem 4.1 <sub>p</sub>roves the u<sub>pp</sub>er half of (58).

F<sub>or</sub> th<sub>e converse, c</sub>h<sub>oose a cons</sub>t<sub>an</sub>t

$$
c _ { B , s } ^ { \star } \leq \frac { 1 } { 4 } a _ { B } ^ { - 1 / ( 2 s ) }\tag{135}
$$

an<sup>d</sup> <sub>p</sub>ut $J = \bigl \lfloor c _ { B , s } ^ { \star } M \bigr \rfloor$ <sub>.</sub> F<sub>o</sub>r � l<sub>a</sub>r<sub>ge</sub>r th<sub>a</sub>n <sub>a co</sub>n<sub>s</sub>t<sub>a</sub>nt<sub>,</sub> $J \geq 1$ . <sup>F</sup>or ever<sub>y</sub> $j \leq J ,$

$$
T r _ { j } ^ { 2 } = ( M / j ) ^ { 2 s } \geq ( c _ { B , s } ^ { \star } ) ^ { - 2 s } \geq a _ { B } .\tag{136}
$$

The second condition in (57), with a suficientl<sub>y</sub> small $_ { c _ { B , s } }$ <sub>,</sub> i<sub>mp</sub>li<sub>es</sub> b<sub>o</sub>th $J \le T / 2$ <sub>an</sub>d th<sub>e</sub> di<sub>mens</sub>i<sub>on con</sub>diti<sub>on</sub> i<sub>n</sub> C<sub>oro</sub>ll<sub>ary</sub> 4<sub>.</sub>6<sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
\mathcal { R } _ { T } ( r ) \geq c _ { B } \sum _ { j = 1 } ^ { J } \log \left( 1 + ( M / j ) ^ { 2 s } \right) .\tag{137}
$$

F<sub>or</sub> $\lceil J / 2 \rceil \leq j \leq J ;$ <sub>,</sub> th<sub>e ra</sub>ti<sub>o</sub> $M / j$ i<sub>s a</sub>t l<sub>eas</sub>t $1 / c _ { B , s } ^ { \star } ,$ <sub>so</sub> <sub>every</sub> t<sub>erm</sub> i<sub>n</sub> thi<sub>s</sub> <sub>su</sub>b<sub>-</sub>bl<sub>oc</sub>k i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>e</sub>l<sub>ow</sub> b<sub>y</sub> <sub>a</sub> <sub>pos</sub>iti<sub>ve cons</sub>t<sub>an</sub>t d<sub>epen</sub>di<sub>ng on</sub>l<sub>y on</sub> $B , s$ <sub>.</sub> Th<sub>ere are</sub> $\Theta _ { B , s } ( M )$ <sub>suc</sub>h i<sub>n</sub>di<sub>ces w</sub>hi<sub>c</sub>h <sub>roves</sub> th<sub>e</sub> l<sub>ower</sub> h<sub>a</sub>lf <sub>o</sub>f (58). For fixed � and $s > 1$

$$
M ^ { 2 } \log ( 2 T ) = A ^ { 2 / s } T ^ { 1 / s } \log ( 2 T ) = o ( T ) ,\tag{138}
$$

so (59) follows.

## E.4 Memory-profile sum bounds in Lemma E.1

Lemma E.1 (Exponential and polynomial tail sums). Let $1 \leq h \leq T / 4$

1. ${ \it I f r } _ { j } = A e ^ { - \alpha j }$ with $\alpha > 0 ,$ , then

$$
c _ { \alpha } T A ^ { 2 } e ^ { - 2 \alpha h } \leq \sum _ { j > h } n _ { T , j } r _ { j } ^ { 2 } \leq C _ { \alpha } T A ^ { 2 } e ^ { - 2 \alpha h } .\tag{139}
$$

2. I $f r _ { j } = A j ^ { - s }$ with $s > 1 / 2$ , then

$$
c _ { s } T A ^ { 2 } h ^ { 1 - 2 s } \leq \sum _ { j > h } n _ { T , j } r _ { j } ^ { 2 } \leq C _ { s } T A ^ { 2 } h ^ { 1 - 2 s } .\tag{140}
$$

Proof. For the exponential upper bound,

$$
\sum _ { j > h } n _ { T , j } A ^ { 2 } e ^ { - 2 \alpha j } \leq T A ^ { 2 } \frac { e ^ { - 2 \alpha ( h + 1 ) } } { 1 - e ^ { - 2 \alpha } } .\tag{141}
$$

Th<sub>e</sub> fi<sub>rs</sub>t <sub>om</sub>itt<sub>e</sub>d t<sub>erm</sub> h<sub>as</sub> $n _ { T , h + 1 } = T - h \geq 3 T / 4$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>w</sub>ith <sub>a</sub> <sub>cons</sub>t<sub>an</sub>t d<sub>epen</sub>di<sub>ng</sub> <sub>on</sub> �.

F<sub>or</sub> th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>upper</sub> b<sub>oun</sub>d<sub>,</sub> th<sub>e</sub> i<sub>n</sub>t<sub>egra</sub>l t<sub>es</sub>t <sub>g</sub>i<sub>ves</sub>

$$
\sum _ { j > h } n _ { T , j } A ^ { 2 } j ^ { - 2 s } \leq T A ^ { 2 } \sum _ { j > h } j ^ { - 2 s } \leq C _ { s } T A ^ { 2 } h ^ { 1 - 2 s } .\tag{142}
$$

F<sub>or</sub> th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>,</sub> <sub>sum</sub> <sub>on</sub>l<sub>y</sub> <sub>over</sub> $j = h + 1 , \ldots , 2 h$ <sub>.</sub> Si<sub>nce</sub> $2 h \leq T / 2$ <sub>, every suc</sub>h l<sub>ag</sub> h<sub>as</sub> $n _ { T , j } \ge T / 2$ <sub>,</sub> <sub>an</sub>d <sub>every</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $j ^ { - 2 \bar { s } } \ge ( 2 h ) ^ { - 2 s }$ . The ℎ terms <sub>g</sub>ive (140). □

Fi<sub>na</sub>ll<sub>y,</sub> <sub>op</sub>ti<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>e</sub> h<sub>ar</sub>d<sub>-</sub>t<sub>runca</sub>ti<sub>on</sub> <sub>express</sub>i<sub>on</sub> i<sub>n</sub> th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>case</sub> i<sub>s</sub> <sub>e</sub>l<sub>emen</sub>t<sub>ary.</sub> B<sub>a</sub>l<sub>anc</sub>i<sub>ng</sub> ℎ l<sub>og</sub> � <sub>an</sub>d $T A ^ { 2 } h ^ { 1 - 2 s }$ <sub>g</sub><sup>i</sup>ves $h \asymp ( T / \log T ) ^ { 1 / ( 2 s ) }$ (with constants de<sub>p</sub>endin<sub>g</sub> on $A , s )$ <sub>.</sub> F<sub>or</sub> fi<sub>xe</sub>d �<sub>,</sub> <sub>�</sub> <sub>an</sub>d <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge</sub> �<sub>,</sub> thi<sub>s</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> li<sub>es</sub> b<sub>e</sub>l<sub>ow</sub> $T / 4$ <sub>,</sub> <sub>an</sub>d <sub>roun</sub>di<sub>ng</sub> it t<sub>o</sub> <sub>an</sub> i<sub>n</sub>t<sub>eger</sub> <sub>c</sub>h<sub>anges</sub> <sub>on</sub>l<sub>y</sub> <sub>cons</sub>t<sub>an</sub>t<sub>s.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> (61).

## Appendix F Proof of Theorem 5.1

W<sub>e prove a</sub> d<sub>es</sub>i<sub>gn-sens</sub>iti<sub>ve</sub> ONS i<sub>nequa</sub>lit<sub>y an</sub>d th<sub>en app</sub>l<sub>y</sub> it t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e orac</sub>l<sub>e</sub> b<sub>oun</sub>d i<sub>n</sub> Th<sub>eorem</sub> 5<sub>.</sub>1<sub>.</sub> Th<sub>e</sub> <sub>spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>argumen</sub>t k<sub>eeps</sub> th<sub>e</sub> d<sub>epen</sub>d<sub>ence on</sub> th<sub>e an</sub>i<sub>so</sub>t<sub>rop</sub>i<sub>c</sub> d<sub>es</sub>i<sub>gn exp</sub>li<sub>c</sub>it<sub>.</sub>

## F.1 Auxiliary bounded-logit ONS lemma for Theorem 5.1

L<sub>e</sub>t $K \subset \mathbb { R } ^ { d }$ b<sub>e c</sub>l<sub>ose</sub>d <sub>an</sub>d <sub>convex,</sub> l<sub>e</sub>t $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { d }$ <sub>,</sub> <sub>an</sub>d l<sub>e</sub>t

$$
\ell _ { t } ( u ) = \psi ( x _ { t } ^ { \top } u ) - y _ { t } x _ { t } ^ { \top } u , \qquad y _ { t } \in \{ 0 , 1 \} .\tag{143}
$$

A<sub>ssume</sub>

$$
| x _ { t } ^ { \top } u | \leq B \qquad { \mathrm { f o r e v e r y ~ } } u \in K { \mathrm { ~ a n d ~ e v e r y ~ } } t .\tag{144}
$$

L<sub>e</sub>t ${ { g } _ { t } } = \nabla { { \ell } _ { t } } \left( { { u } _ { t } } \right)$ <sub>an</sub>d <sub>se</sub>t $\beta _ { B } = \kappa _ { B }$ <sub>.</sub> St<sub>ar</sub>ti<sub>ng</sub> f<sub>rom</sub> $H _ { 0 } = I _ { d }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
H _ { t } = H _ { t - 1 } + \beta _ { B } g _ { t } g _ { t } ^ { \top } , \qquad u _ { t + 1 } = \Pi _ { K } ^ { H _ { t } } ( u _ { t } - H _ { t } ^ { - 1 } g _ { t } ) ,\tag{145}
$$

<sub>w</sub>h<sub>ere</sub> $\Pi _ { K } ^ { H } ( \nu )$ <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> $\| u - \nu \| _ { H } ^ { 2 } = ( u - \nu ) ^ { \top } H ( u - \nu )$ over $u \in K$

Lemma F.1 (Design-sensitive ONS). For every $u \in K _ { : }$

$$
\sum _ { t = 1 } ^ { T } \bigl ( \ell _ { t } ( u _ { t } ) - \ell _ { t } ( u ) \bigr ) \leq \frac { 1 } { 2 } \| u _ { 1 } - u \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 \beta _ { B } } \log \operatorname* { d e t } \biggl ( I _ { d } + \beta _ { B } \sum _ { t = 1 } ^ { T } g _ { t } g _ { t } ^ { \intercal } \biggr ) .\tag{146}
$$

Since $g _ { t } = ( \sigma ( x _ { t } ^ { \top } u _ { t } ) - y _ { t } ) x _ { t }$ and $\beta _ { B } \leq 1$

$$
\sum _ { t = 1 } ^ { T } \bigl ( \ell _ { t } ( u _ { t } ) - \ell _ { t } ( u ) \bigr ) \leq \frac { 1 } { 2 } \| u _ { 1 } - u \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 \beta _ { B } } \log \operatorname* { d e t } \biggl ( I _ { d } + \sum _ { t = 1 } ^ { T } x _ { t } x _ { t } ^ { \top } \biggr ) .\tag{147}
$$

Proof. Fix $u , \nu \in K$ . Ta<sub>y</sub>lor’s theorem and (144) <sub>g</sub>ive

$$
\ell _ { t } ( \nu ) \geq \ell _ { t } ( u ) + \nabla \ell _ { t } ( u ) ^ { \top } ( \nu - u ) + \frac { \kappa _ { B } } { 2 } \big ( x _ { t } ^ { \top } ( \nu - u ) \big ) ^ { 2 } .\tag{148}
$$

B<sub>ecause</sub> $\nabla \ell _ { t } ( u ) = ( \sigma ( x _ { t } ^ { \top } u ) - y _ { t } ) x _ { t }$ <sub>an</sub>d th<sub>e</sub> <sub>sca</sub>l<sub>ar</sub> <sub>mu</sub>lti<sub>p</sub>li<sub>er</sub> h<sub>as</sub> <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> <sub>a</sub>t <sub>mos</sub>t <sub>one,</sub>

$$
\begin{array} { r } { \left( x _ { t } ^ { \top } ( \nu - u ) \right) ^ { 2 } \geq \left( \nabla \ell _ { t } ( u ) ^ { \top } ( \nu - u ) \right) ^ { 2 } . } \end{array}\tag{149}
$$

A<sub>pp</sub>l<sub>y</sub> (148) with $\boldsymbol { u } = \boldsymbol { u } _ { t }$ <sub>an</sub>d $\nu = u$ t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\ell _ { t } ( u _ { t } ) - \ell _ { t } ( u ) \leq g _ { t } ^ { \top } ( u _ { t } - u ) - \frac { \beta _ { B } } { 2 } \big ( g _ { t } ^ { \top } ( u _ { t } - u ) \big ) ^ { 2 } .\tag{150}
$$

Metric <sub>p</sub>rojection is nonex<sub>p</sub>ansive relative to ever<sub>y</sub> <sub>p</sub>oint in �, so (145) im<sub>p</sub>lies

$$
\begin{array} { r l } & { \left\| u _ { t + 1 } - u \right\| _ { H _ { t } } ^ { 2 } \leq \left\| u _ { t } - H _ { t } ^ { - 1 } g _ { t } - u \right\| _ { H _ { t } } ^ { 2 } } \\ & { \qquad = \left\| u _ { t } - u \right\| _ { H _ { t } } ^ { 2 } - 2 g _ { t } ^ { \top } ( u _ { t } - u ) + g _ { t } ^ { \top } H _ { t } ^ { - 1 } g _ { t } . } \end{array}\tag{151}
$$

R<sub>ea</sub>rr<sub>a</sub>n<sub>g</sub>in<sub>g,</sub> <sub>a</sub>nd <sub>us</sub>in<sub>g</sub>

$$
\Vert u _ { t } - u \Vert _ { H _ { t } } ^ { 2 } = \Vert u _ { t } - u \Vert _ { H _ { t - 1 } } ^ { 2 } + \beta _ { B } \big ( g _ { t } ^ { \top } ( u _ { t } - u ) \big ) ^ { 2 } ,\tag{152}
$$

w<sup>e</sup> g<sup>et</sup>

$$
\begin{array} { r l } & { g _ { t } ^ { \top } ( u _ { t } - u ) - \displaystyle \frac { \beta _ { B } } { 2 } \big ( g _ { t } ^ { \top } ( u _ { t } - u ) \big ) ^ { 2 } } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } \left( \| u _ { t } - u \| _ { H _ { t - 1 } } ^ { 2 } - \| u _ { t + 1 } - u \| _ { H _ { t } } ^ { 2 } \right) + \frac { 1 } { 2 } g _ { t } ^ { \top } H _ { t } ^ { - 1 } g _ { t } . } \end{array}\tag{153}
$$

Summin<sub>g</sub> (150) and (153) telesco<sub>p</sub>es the <sub>p</sub>otential and <sub>y</sub>ields

$$
\sum _ { t = 1 } ^ { T } ( \ell _ { t } ( u _ { t } ) - \ell _ { t } ( u ) ) \leq \frac { 1 } { 2 } \| u _ { 1 } - u \| _ { H _ { 0 } } ^ { 2 } + \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } g _ { t } ^ { \top } H _ { t } ^ { - 1 } g _ { t } .\tag{154}
$$

L<sub>e</sub>t $\xi _ { t } = \beta _ { B } g _ { t } ^ { \top } H _ { t - 1 } ^ { - 1 } g _ { t }$ <sub>.</sub> Th<sub>e</sub> Sh<sub>erman–</sub>M<sub>orr</sub>i<sub>son an</sub>d <sub>ma</sub>t<sub>r</sub>i<sub>x-</sub>d<sub>e</sub>t<sub>erm</sub>i<sub>nan</sub>t l<sub>emmas g</sub>i<sub>ve</sub>

$$
\beta _ { B } g _ { t } ^ { \top } H _ { t } ^ { - 1 } g _ { t } = \frac { \xi _ { t } } { 1 + \xi _ { t } } \leq \log ( 1 + \xi _ { t } ) = \log \frac { \operatorname* { d e t } H _ { t } } { \operatorname* { d e t } H _ { t - 1 } } .\tag{155}
$$

Summin<sub>g</sub> (155) in (154) <sub>p</sub>roves (146).

Fi<sub>na</sub>ll<sub>y,</sub> ${ { g } _ { t } } { { g } _ { t } } ^ { \top } \preceq { { x } _ { t } } { { x } _ { t } } ^ { \top }$ <sub>,</sub> <sub>an</sub>d $0 < \beta _ { B } \le 1 / 4 < 1$ <sub>.</sub> M<sub>ono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y o</sub>f $M \mapsto \log \operatorname* { d e t } ( I + M )$ <sub>on</sub> th<sub>e pos</sub>iti<sub>ve-</sub> semidefinite cone <sub>g</sub>ives (147). □

## F.2 Proof of Theorem 5.1: Application to the profile-scaled predictor

T<sub>a</sub>k<sub>e</sub> $K = [ - 1 , 1 ] ^ { h } , u _ { 1 } = 0$ , and the features (70). The envelo<sub>p</sub>e condition <sub>g</sub>ives

$$
{ | ( x _ { t } ^ { ( h ) } ) ^ { \top } u | } \leq \sum _ { j = 1 } ^ { h } r _ { j } \leq B\tag{156}
$$

<sup>f</sup>or ever<sub>y</sub> $u \in K$ <sub>.</sub> F<sub>or</sub> th<sub>e norma</sub>li<sub>ze</sub>d t<sub>runca</sub>t<sub>e</sub>d <sub>compara</sub>t<sub>or</sub>

$$
u _ { j } ^ { \star } = \theta _ { j } / r _ { j } , \qquad j \le h ,\tag{157}
$$

<sub>w</sub>ith th<sub>e conven</sub>ti<sub>on</sub> $u _ { j } ^ { \star } = 0$ <sub>w</sub>h<sub>en</sub> $r _ { j } = 0$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $\begin{array} { r } { \| u ^ { \star } \| _ { 2 } ^ { 2 } \leq h } \end{array}$ <sub>.</sub> Th<sub>us</sub> L<sub>e</sub>mm<sub>a</sub> F<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub>

$$
\sum _ { t = 1 } ^ { T } \bigl ( \ell _ { t } ( u _ { t } ) - \ell _ { t } ( u ^ { \star } ) \bigr ) \leq C _ { B } \left[ h + \log \operatorname* { d e t } \biggl ( I _ { h } + \sum _ { t = 1 } ^ { T } x _ { t } ^ { ( h ) } ( x _ { t } ^ { ( h ) } ) ^ { \top } \biggr ) \right] .\tag{158}
$$

Hadamard’s inequality bounds the determinant by the product of the diagonal entries. Coordinate � is nonzero on exact<sup>l</sup><sub>y</sub> $n _ { T , j }$ <sub>roun</sub>d<sub>s</sub> <sub>an</sub>d h<sub>as</sub> <sub>square</sub>d <sub>va</sub>l<sub>ue</sub> $r _ { j } ^ { 2 } ,$ so

$$
\log \operatorname* { d e t } \left( I _ { h } + \sum _ { t = 1 } ^ { T } x _ { t } ^ { ( h ) } ( x _ { t } ^ { ( h ) } ) ^ { \top } \right) \leq \sum _ { j = 1 } ^ { h } \log ( 1 + n _ { T , j } r _ { j } ^ { 2 } ) .\tag{159}
$$

Th<sub>e co</sub>m<sub>pa</sub>r<sub>a</sub>t<sub>o</sub>r $u ^ { \star }$ <sub>pre</sub>di<sub>c</sub>t<sub>s accor</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> t<sub>runca</sub>t<sub>e</sub>d <sub>source</sub> $P _ { \theta ^ { ( h ) } , T }$ . Takin<sub>g</sub> ex<sub>p</sub>ectations in (158), <sub>su</sub>bt<sub>rac</sub>ti<sub>ng</sub> th<sub>e</sub> B<sub>ayes</sub> l<sub>oss</sub> <sub>un</sub>d<sub>er</sub> $P _ { \theta , T }$ , and usin<sub>g</sub> the u<sub>pp</sub>er half of (27) <sub>y</sub>ields

$$
\mathrm { R e g } _ { T } ( \mathrm { O N S } _ { r } ; P _ { \theta , T } ) \leq C _ { B } \left[ h + \sum _ { j = 1 } ^ { h } \log ( 1 + n _ { T , j } r _ { j } ^ { 2 } ) \right] + \frac { 1 } { 8 } \sum _ { j > h } n _ { T , j } \theta _ { j } ^ { 2 } .\tag{160}
$$

This is (72). The reduction to $C _ { B } \Gamma _ { T } ( r )$ i<sub>s g</sub>i<sub>ven</sub> i<sub>n</sub> th<sub>e proo</sub>f i<sub>n</sub> th<sub>e ma</sub>i<sub>n</sub> t<sub>ex</sub>t<sub>.</sub>

## Appendix G Proof of Theorem 4.10

W<sub>e prove</sub> th<sub>e re</sub>d<sub>un</sub>d<sub>ancy</sub> b<sub>oun</sub>d<sub>s</sub> i<sub>n</sub> Th<sub>eorem</sub> 4<sub>.</sub>10<sub>.</sub>

## G.1 Upper bound in Theorem 4.10

F<sub>or</sub> $\theta = a r$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>t<sub>a</sub>bl<sub>e</sub> <sub>sca</sub>l<sub>ar</sub> f<sub>ea</sub>t<sub>ure</sub>

$$
z _ { t } = \sum _ { j = 1 } ^ { t } r _ { j } U _ { t + 1 - j } .\tag{161}
$$

Th<sub>en</sub> th<sub>e mar</sub>k l<sub>og</sub>it i<sub>s</sub> $a z _ { t } , | z _ { t } | \leq B$ <sub>,</sub> <sub>an</sub>d

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } z _ { t } ^ { 2 } = \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { t } r _ { j } ^ { 2 } = \sum _ { j = 1 } ^ { T } n _ { T , j } r _ { j } ^ { 2 } = I _ { T } ( r ) .\tag{162}
$$

B<sub>y</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l l<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> KL <sub>upper</sub> b<sub>oun</sub>d<sub>,</sub> f<sub>or</sub> $a , b \in [ - 1 , 1 ]$ ,

$$
\mathrm { K L } ( P _ { a r , T } | | P _ { b r , T } ) \leq \frac { 1 } { 8 } ( a - b ) ^ { 2 } I _ { T } ( r ) .\tag{163}
$$

A<sub>pp</sub>l<sub>y</sub> th<sub>e var</sub>i<sub>a</sub>ti<sub>ona</sub>l<sub>-m</sub>i<sub>x</sub>t<sub>ure cons</sub>t<sub>ruc</sub>ti<sub>on</sub> f<sub>rom</sub> A<sub>ppen</sub>di<sub>x</sub> B i<sub>n one</sub> di<sub>mens</sub>i<sub>on, w</sub>ith th<sub>e un</sub>if<sub>orm pr</sub>i<sub>or on</sub> $[ - 1 , 1 ]$ <sub>.</sub> If $I _ { T } ( r ) \leq 1$ <sub>, use</sub> th<sub>e w</sub>h<sub>o</sub>l<sub>e pr</sub>i<sub>or an</sub>d <sub>pay a</sub>t <sub>mos</sub>t $I _ { T } ( r ) / 2$ <sub>.</sub> If $I _ { T } ( r ) > 1$ <sub>,</sub> l<sub>oca</sub>li<sub>ze</sub> t<sub>o an</sub> i<sub>n</sub>t<sub>erva</sub> <sub>o</sub>f l<sub>eng</sub>th $I _ { T } ( r ) ^ { - 1 / 2 }$ adjacent to the true $^ { a ; }$ th<sub>e pa</sub>i<sub>rw</sub>i<sub>se cos</sub>t i<sub>s a</sub>t <sub>mos</sub>t $1 / 8$ <sub>an</sub>d th<sub>e</sub> l<sub>oca</sub>li<sub>za</sub>ti<sub>on cos</sub>t i<sub>s</sub> $\frac { 1 } { 2 }$ <sup>l</sup>o<sub>g</sub> $I _ { T } ( r ) + \log 2$ <sub>.</sub> I<sub>n</sub> b<sub>o</sub>th <sub>cases,</sub>

$$
\mathcal { R } _ { T } ^ { \mathrm { r a y } } ( r ) \leq \frac { 1 } { 2 } \log ( 1 + I _ { T } ( r ) ) + C .\tag{164}
$$

## G.2 Empirical feature-energy bound for Theorem 4.10

L<sub>e</sub>t $\mathcal { F } _ { t - 1 } = \sigma ( U _ { 1 } , \dots , U _ { t - 1 } )$ . <sup>D</sup>ecom<sub>p</sub>ose

$$
z _ { t } = w _ { t } + r _ { 1 } U _ { t } , \qquad w _ { t } = \sum _ { j = 2 } ^ { t } r _ { j } U _ { t + 1 - j } ,\tag{165}
$$

<sub>w</sub>h<sub>ere</sub> $w _ { t }$ i<sub>s</sub> $\mathcal { F } _ { t - 1 }$ <sub>-measura</sub>bl<sub>e.</sub> F<sub>or every rea</sub>l $w ,$

$$
\operatorname* { m a x } \{ | w + r _ { 1 } | , | w - r _ { 1 } | \} \geq r _ { 1 } .\tag{166}
$$

Th<sub>ere</sub>f<sub>ore,</sub> <sub>w</sub>ith $I _ { t } = \mathbf { 1 } \{ | z _ { t } | \geq r _ { 1 } \}$

$$
\mathbb { E } [ I _ { t } \mid { \mathcal { F } } _ { t - 1 } ] \geq { \frac { 1 } { 2 } } .\tag{167}
$$

Th<sub>e</sub> dif<sub>erences</sub> $I _ { t } - \mathbb { E } [ I _ { t } \mid { \mathcal { F } } _ { t - 1 } ]$ f<sub>orm</sub> <sub>a</sub> <sub>mar</sub>ti<sub>nga</sub>l<sub>e-</sub>dif<sub>erence</sub> <sub>sequence</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> <sub>one</sub> i<sub>n</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>va</sub>l<sub>ue.</sub> A<sub>zuma–</sub>H<sub>oe</sub>fdi<sub>ng g</sub>i<sub>ves</sub>

$$
\mathbb { P } \left( \sum _ { t = 1 } ^ { T } I _ { t } < \frac { T } { 4 } \right) \le e ^ { - T / 3 2 } .\tag{168}
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>, on an event $\mathcal { H } _ { T }$ <sub>o</sub>f <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $1 - e ^ { - T / 3 2 }$

$$
\sum _ { t = 1 } ^ { T } z _ { t } ^ { 2 } \geq \frac { T r _ { 1 } ^ { 2 } } { 4 } .\tag{169}
$$

## G.3 Lower bound in Theorem 4.10: One-dimensional entropy argument

Pl<sub>ace</sub> th<sub>e</sub> <sub>un</sub>if<sub>orm</sub> <sub>pr</sub>i<sub>or</sub> <sub>on</sub> $a \in \left[ - 1 / 2 , 1 / 2 \right]$ <sub>.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> i<sub>npu</sub>t<sub>s,</sub> l<sub>e</sub>t

$$
\ell _ { T } ( a ) = \sum _ { t = 1 } ^ { T } \left[ \psi ( a z _ { t } ) - Y _ { t + 1 } a z _ { t } \right]\tag{170}
$$

b<sub>e</sub> th<sub>e nega</sub>ti<sub>ve</sub> l<sub>og</sub> lik<sub>e</sub>lih<sub>oo</sub>d<sub>, an</sub>d l<sub>e</sub>t $\widehat { a }$ <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> it <sub>over</sub> $[ - 1 / 2 , 1 / 2 ]$ <sub>.</sub> Si<sub>nce</sub> $| a z _ { t } | \le B / 2$ th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>,</sub>

$$
\ell _ { T } ^ { \prime \prime } ( a ) \geq \kappa _ { B / 2 } \sum _ { t = 1 } ^ { T } z _ { t } ^ { 2 } .\tag{171}
$$

W<sub>r</sub>it<sub>e</sub> $\begin{array} { r } { S _ { T } = \sum _ { t = 1 } ^ { T } z _ { t } ^ { 2 } } \end{array}$ <sub>.</sub> Si<sub>nce</sub> b<sub>o</sub>th <sub>� an</sub>d $\widehat { a }$ li<sub>e</sub> i<sub>n</sub> th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>n</sub>t<sub>erva</sub>l <sub>an</sub>d $\ell _ { T } ( \widehat { a } ) \leq \ell _ { T } ( a )$ , <sup>stron</sup>g <sup>con</sup>v<sup>exit</sup>y <sub>g</sub><sup>i</sup>ves

$$
0 \geq \ell _ { T } ^ { \prime } ( a ) ( \widehat { a } - a ) + \frac { \kappa _ { B / 2 } S _ { T } } { 2 } ( \widehat { a } - a ) ^ { 2 } .
$$

Th<sub>ere</sub>f<sub>ore, on</sub> $\mathcal { H } _ { T }$

$$
| \widehat { a } - a | \leq \frac { 2 | \ell _ { T } ^ { \prime } ( a ) | } { \kappa _ { B / 2 } S _ { T } } .\tag{172}
$$

C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> i<sub>npu</sub>t<sub>s an</sub>d <sub>�,</sub> th<sub>e score</sub> i<sub>s cen</sub>t<sub>ere</sub>d <sub>an</sub>d

$$
\mathbb { E } [ \ell _ { T } ^ { \prime } ( a ) ^ { 2 } \mid U _ { 1 : T } , a ] \leq \frac { 1 } { 4 } \sum _ { t = 1 } ^ { T } z _ { t } ^ { 2 } .\tag{173}
$$

B<sub>ecause</sub> $\mathcal { H } _ { T }$ is determined b<sub>y</sub> the in<sub>p</sub>uts, (172) and (173) im<sub>p</sub>l<sub>y</sub>

$$
\begin{array} { r l } & { \mathbb { E } \Big [ ( \widehat { a } - a ) ^ { 2 } \mathbf { 1 } _ { \mathcal { H } _ { T } } \mid U _ { 1 : T } , a \Big ] \leq \frac { 4 } { \kappa _ { B / 2 } ^ { 2 } S _ { T } ^ { 2 } } \mathbb { E } [ \ell _ { T } ^ { \prime } ( a ) ^ { 2 } \mid U _ { 1 : T } , a ] \mathbf { 1 } _ { \mathcal { H } _ { T } } } \\ & { \qquad \leq \frac { 1 } { \kappa _ { B / 2 } ^ { 2 } S _ { T } } \mathbf { 1 } _ { \mathcal { H } _ { T } } \leq \frac { 4 } { \kappa _ { B / 2 } ^ { 2 } T r _ { 1 } ^ { 2 } } . } \end{array}
$$

T<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith th<sub>e</sub> di<sub>ame</sub>t<sub>er</sub> b<sub>oun</sub>d $| { \widehat { a } } - a | \leq 1$ on $\mathcal { H } _ { T } ^ { c }$ <sub>,</sub> thi<sub>s</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\mathbb { E } ( \widehat { a } - a ) ^ { 2 } \leq \frac { C _ { B } } { T r _ { 1 } ^ { 2 } } + e ^ { - T / 3 2 } \leq \frac { C _ { B } } { T r _ { 1 } ^ { 2 } } .\tag{174}
$$

Th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> <sub>uses</sub> $r _ { 1 } \leq B$ <sub>an</sub>d th<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>ness o</sub>f $T e ^ { - T / 3 2 }$

Th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> i<sub>n</sub>t<sub>erva</sub>l h<sub>as</sub> l<sub>eng</sub>th <sub>one,</sub> <sub>so</sub> $\boldsymbol { h } ( \boldsymbol { a } ) = 0$ <sub>.</sub> B<sub>y</sub> d<sub>a</sub>t<sub>a process</sub>i<sub>ng an</sub>d th<sub>e sca</sub>l<sub>ar</sub> G<sub>auss</sub>i<sub>an max</sub>i<sub>mum-</sub> <sup>entro</sup>py <sup>ine</sup>qu<sup>alit</sup>y,

$$
\begin{array} { r l } & { I ( a ; X _ { 1 : T + 1 } ) \geq I ( a ; \widehat { a } ) } \\ & { \qquad = h ( a ) - h ( a \mid \widehat { a } ) } \\ & { \qquad \geq - h ( a - \widehat { a } ) } \\ & { \qquad \geq - \displaystyle \frac { 1 } { 2 } \log \left( 2 \pi e \mathbb { E } ( a - \widehat { a } ) ^ { 2 } \right) } \\ & { \qquad \geq \displaystyle \frac { 1 } { 2 } \log ( T r _ { 1 } ^ { 2 } ) - C _ { B } . } \end{array}\tag{175}
$$

The redundanc<sub>y</sub>–ca<sub>p</sub>acit<sub>y</sub> inequalit<sub>y</sub> Lemma D.1, to<sub>g</sub>ether with nonne<sub>g</sub>ativit<sub>y</sub>, <sub>p</sub>roves the lower half of (65).

Fi<sub>na</sub>ll<sub>y,</sub> f<sub>or a</sub> fi<sub>xe</sub>d <sub>summa</sub>bl<sub>e pro</sub>fil<sub>e w</sub>ith $r _ { 1 } > 0$

$$
T r _ { 1 } ^ { 2 } \leq I _ { T } ( r ) \leq T \sum _ { j \geq 1 } r _ { j } ^ { 2 } \leq T B ^ { 2 } .\tag{176}
$$

The two sides of (65) are therefore both of order lo<sub>g</sub> �, <sub>p</sub>rovin<sub>g</sub> (66).

## References

Atteson, K. (1999). “The Asymptotic Redundancy of Bayes Rules for Markov Chains”. In: IEEE Transactions on Information Theory 45.6, pp. 2104–2109. doi: 10.1109/18.782131.

Beylkin, G. and L. Monzon (2010). “Approximation by Exponential Sums Revisited”. In:´ Applied and Computational Harmonic Analysis 28.2, pp. 131–149. doi: 10.1016/j.acha.2009.08.011.

Clarke, B. S. and A. R. Barron (1990). “Information-Theoretic Asymptotics of Bayes Methods”. In: IEEE Transactions on Information Theory 36.3, pp. 453–471. doi: 10.1109/18.54897.

Crutchfield, J. P. and K. Young (1989). “Inferring Statistical Complexity”. In: Physical Review Letters 63.2, <sub>pp</sub>. 105–108. doi: 10.1103/PhysRevLett.63.105.

Drmota, M., P. Jacquet, C. Wu, and W. Sz<sub>p</sub>ankowski (2026). “Phase Transition of Re<sub>g</sub>ret for Lo<sub>g</sub>istic Regression with Large Weights”. In: Proceedings ofthe 37th International Conference on Algorithmic Learning Theory. Vol. 313. Proceedings of Machine Learning Research, pp. 1–28.

Fokianos, K. and L. Truquet (2019). “On Categorical Time Series Models with Covariates”. In: Stochastic Processes and their Applications 129.9, pp. 3446–3462. doi: 10.1016/j.spa.2018.09.012.

Foster, D. J., S. Kale, H. Luo, M. Mohri, and K. Sridharan (2018). “Lo istic Re ression: The Im ortance of Being Improper”. In: Proceedings ofthe 31st Conference on Learning Theory. Vol. 75. Proceedings of M<sub>ac</sub>hi<sub>ne</sub> L<sub>earn</sub>i<sub>ng</sub> R<sub>esearc</sub>h<sub>,</sub> <sub>pp.</sub> 167<sub>–</sub>208<sub>.</sub>

Goldenshlu<sub>g</sub>er, A. and A. Zeevi (2001). “Nonas<sub>y</sub>m<sub>p</sub>totic Bounds for Autore<sub>g</sub>ressive Time Series Modelin<sub>g</sub>”. In: The Annals of Statistics 29.2, <sub>pp</sub>. 417–444. doi: 10.1214/aos/1009210547.

Han, Y., S. Jana, and Y. Wu (2023). “O<sub>p</sub>timal Prediction of Markov Chains With and Without S<sub>p</sub>ectral Ga<sub>p</sub>”. In: IEEE Transactions on Information Theory 69.6, pp. 3920–3959. doi: 10.1109/TIT.2023.3239508.

Han, Y., T. Jian<sub>g</sub>, and Y. Wu (2024). “Prediction from Com<sub>p</sub>ression for Models with Infinite Memor<sub>y</sub>, with Applications to Hidden Markov and Renewal Processes”. In: Proceedings of the 37th Conference on Learning Theory. Vol. 247. Proceedings of Machine Learning Research, pp. 2270–2307.

Hazan, E., A. A<sub>g</sub>arwal, and S. Kale (2007). “Lo<sub>g</sub>arithmic Re<sub>g</sub>ret Al<sub>g</sub>orithms for Online Convex O<sub>p</sub>timization”. In: Machine Learning 69.2–3, <sub>pp</sub>. 169–192. doi: 10.1007/s10994-007-5016-8.

Jacquet, P., G. I. Shamir, and W. Sz<sub>p</sub>ankowski (2021). “Precise Minimax Re<sub>g</sub>ret for Lo<sub>g</sub>istic Re<sub>g</sub>ression with Categorical Feature Values”. In: Proceedings ofthe 32nd International Conference on Algorithmic Learning Theory. Vol. 132. Proceedings of Machine Learning Research, pp. 755–771.

Marsden, A. and E. Hazan (2026). The Power of Second Order Methods for Sequence Preconditioning. arXiv: 2605.08390 [cs.LG].

Marzen S. E. and J. P. Crutchfield (2016). “Predictive Rate–Distortion for Infinite-Order Markov Processes”. In: Journal of Statistical Physics 163, pp. 1312–1338. doi: 10.1007/s10955-016-1520-1.

Merhav, N. and M. Feder (1998). “Universal Prediction”. In: IEEE Transactions on Information Theory 44.6, <sub>pp</sub>. 2124–2147. doi: 10.1109/18.720534.

Rissanen, J. (1996). “Fisher Information and Stochastic Complexity”. In: IEEE Transactions on Information Theory 42.1, <sub>pp</sub>. 40–47. doi: 10.1109/18.481776.

Shalizi, C. R. and J. P. Crutchfield (2001). “Com<sub>p</sub>utational Mechanics: Pattern and Prediction, Structure and Simplicity”. In: Journal ofStatistical Physics 104, pp. 817–879. doi: 10.1023/A:1010388907793.

Shamir, G. I. (2020). “Logistic Regression Regret: What’s the Catch?” In: Proceedings ofthe 33rd Conference on Learning Theory. Vol. 125. Proceedings of Machine Learning Research, pp. 3296–3319.

Shtar’kov, Y. M. (1987). “Universal sequential coding of single messages”. In: Problemy Peredachi Informatsii 23<sub>.</sub>3<sub>,</sub> <sub>pp.</sub> 3<sub>–</sub>17<sub>.</sub>

Willems, F. M. J., Y. M. Shtarkov, and T. J. Tjalkens (1995). “The Context-Tree Wei<sub>g</sub>htin<sub>g</sub> Method: Basic Properties”. In: IEEE Transactions on Information Theory 41.3, pp. 653–664. doi: 10.1109/18.382012.

Wu, C., M. Hosseini, and N. Santhanam (2018). “Redundanc<sub>y</sub> of Unbounded Memor<sub>y</sub> Markov Classes with Continuity Conditions”. In: 2018 IEEE International Symposium on Information Theory, pp. 211–215. doi: 10.1109/ISIT.2018.8437650.