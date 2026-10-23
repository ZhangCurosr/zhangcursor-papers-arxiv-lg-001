# Local Discrimination of Binary Classifiers via Localized AUC Functional

<sub>Wenye</sub> <sub>Ren</sub>∗<sub>,</sub> 1<sub>,</sub> 2<sub>,</sub> <sub>Anna</sub> <sub>Gyorfy-Kerekes</sub> <sub>˝</sub> \*<sub>,</sub> 1<sub>,</sub> 3<sub>,</sub> <sub>Tancredi</sub> <sub>Dorsi</sub>4<sub>,</sub> <sub>and</sub> <sub>Olga</sub> <sub>Demler</sub>1<sub>,</sub> 5

<sup>1</sup> D<sub>epa</sub>rtm<sub>e</sub>nt <sub>o</sub>f C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub>r S<sub>c</sub>i<sub>e</sub>n<sub>ce,</sub> ETH Z<sub>u</sub>ri<sub>c</sub>h<sub>,</sub> S<sub>w</sub>it<sub>ze</sub>rl<sub>a</sub>nd

<sup>2</sup> D<sub>epa</sub>rtm<sub>e</sub>nt <sub>o</sub>f C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub>r S<sub>c</sub>i<sub>e</sub>n<sub>ce,</sub> T<sub>s</sub>in<sub>g</sub>h<sub>ua</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y,</sub> Chin<sub>a</sub>

<sup>3</sup> C<sub>e</sub>nt<sub>e</sub>r f<sub>o</sub>r L<sub>ea</sub>rnin<sub>g</sub> S<sub>ys</sub>t<sub>e</sub>m<sub>s,</sub> M<sub>ax</sub> Pl<sub>a</sub>n<sub>c</sub>k In<sub>s</sub>tit<sub>u</sub>t<sub>e,</sub> T¨<sub>u</sub>bin<sub>ge</sub>n<sub>,</sub> G<sub>e</sub>rm<sub>a</sub>n<sub>y</sub>

<sup>4</sup> D<sub>epar</sub>t<sub>men</sub>t <sub>o</sub>f M<sub>a</sub>th<sub>ema</sub>ti<sub>cs,</sub> ETH Z<sub>ur</sub>i<sub>c</sub>h<sub>,</sub> S<sub>w</sub>it<sub>zer</sub>l<sub>an</sub>d

<sup>5</sup> M<sub>ass</sub> G<sub>enera</sub>l B<sub>r</sub>i<sub>g</sub>h<sub>am</sub> H<sub>osp</sub>it<sub>a</sub>l<sub>,</sub> H<sub>arvar</sub>d M<sub>e</sub>di<sub>ca</sub>l S<sub>c</sub>h<sub>oo</sub>l<sub>,</sub> B<sub>os</sub>t<sub>on,</sub> MA

O<sub>c</sub>t<sub>o</sub>b<sub>er</sub> 23<sub>,</sub> 2026

## Abstract

## Background

Bi<sub>nary c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s are w</sub>id<sub>e</sub>l<sub>y use</sub>d <sub>across</sub> di<sub>verse</sub> di<sub>sc</sub>i<sub>p</sub>li<sub>nes, w</sub>ith <sub>app</sub>li<sub>ca</sub>ti<sub>ons rang</sub>i<sub>ng</sub> f<sub>rom</sub> di<sub>sease</sub> di<sub>agnos</sub>i<sub>s an</sub>d <sub>r</sub>i<sub>s</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> t<sub>o</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>ki<sub>ng</sub> i<sub>n</sub> fi<sub>nance an</sub>d <sub>eng</sub>i<sub>neer</sub>i<sub>ng.</sub> I<sub>n c</sub>li<sub>n</sub>i<sub>ca</sub>l <sub>me</sub>di<sub>c</sub>i<sub>ne, some o</sub>f th<sub>ese</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> h<sub>ave</sub> b<sub>een</sub> i<sub>ncorpora</sub>t<sub>e</sub>d i<sub>n</sub>t<sub>o</sub> <sub>na</sub>ti<sub>ona</sub>l <sub>an</sub>d i<sub>n</sub>t<sub>erna</sub>ti<sub>ona</sub>l <sub>prac</sub>ti<sub>ce</sub> <sub>gu</sub>id<sub>e</sub>li<sub>nes</sub> <sub>an</sub>d i<sub>n</sub>f<sub>orm</sub> t<sub>rea</sub>t<sub>men</sub>t d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> f<sub>or m</sub>illi<sub>ons o</sub>f <sub>pa</sub>ti<sub>en</sub>t<sub>s wor</sub>ld<sub>w</sub>id<sub>e.</sub> H<sub>owever, mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>ormance can vary su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y across pa</sub>ti<sub>en</sub>t <sub>su</sub>b<sub>groups, an</sub>d t<sub>oo</sub>l<sub>s</sub> t<sub>o sys</sub>t<sub>ema</sub>ti<sub>ca</sub>ll<sub>y</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>ese su</sub>b<sub>groups w</sub>ith<sub>ou</sub>t l<sub>oss o</sub>f <sub>power are</sub> l<sub>ac</sub>ki<sub>ng.</sub>

## Methods

I<sub>n</sub> thi<sub>s</sub> <sub>wor</sub>k<sub>,</sub> <sub>we</sub> d<sub>eve</sub>l<sub>ope</sub>d <sub>a</sub> <sub>me</sub>th<sub>o</sub>d <sub>ca</sub>ll<sub>e</sub>d LOCATOR th<sub>a</sub>t id<sub>en</sub>tifi<sub>es</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow</sub> <sub>mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>ormance</sub> <sub>reg</sub>i<sub>ons</sub> i<sub>n</sub> th<sub>e covar</sub>i<sub>a</sub>t<sub>e space w</sub>ith<sub>ou</sub>t l<sub>oss o</sub>f <sub>power.</sub>

W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce a</sub> l<sub>oca</sub>li<sub>ze</sub>d AUC f<sub>unc</sub>ti<sub>ona</sub>l LAUC<sub>,</sub> th<sub>a</sub>t<sub>,</sub> f<sub>or a re</sub>f<sub>erence r</sub>i<sub>s</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>, es</sub>ti<sub>ma</sub>t<sub>es</sub> AUC f<sub>or</sub> <sub>an</sub> i<sub>n</sub>d<sub>ex par</sub>ti<sub>c</sub>i<sub>pan</sub>t i <sub>us</sub>i<sub>ng a</sub>ll <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s we</sub>i<sub>g</sub>ht<sub>e</sub>d b<sub>y s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> t<sub>o</sub> i<sub>.</sub> B<sub>y us</sub>i<sub>ng</sub> l<sub>oca</sub>li<sub>ze</sub>d AUC f<sub>unc</sub>ti<sub>ona</sub>l <sub>our</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m avo</sub>id<sub>s su</sub>b<sub>se</sub>tti<sub>ng</sub> d<sub>a</sub>t<sub>a a</sub>lt<sub>oge</sub>th<sub>er.</sub> B<sub>ase</sub>li<sub>ne p</sub>h<sub>eno</sub>t<sub>yp</sub>i<sub>c s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> b<sub>e</sub>t<sub>ween</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l<sub>s</sub> i<sub>s</sub> <sub>quan</sub>tifi<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> G<sub>ower</sub> di<sub>s</sub>t<sub>ance.</sub> W<sub>e</sub> th<sub>en</sub> t<sub>ra</sub>i<sub>n</sub> <sub>an</sub> XGB<sub>oos</sub>t <sub>regress</sub>i<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l t<sub>o</sub> <sub>pre</sub>di<sub>c</sub>t th<sub>e</sub> l<sub>og</sub>it<sub>-</sub>t<sub>rans</sub>f<sub>orme</sub>d LAUC <sub>us</sub>i<sub>ng</sub> b<sub>ase</sub>li<sub>ne c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs an</sub>d id<sub>en</sub>tifi<sub>e</sub>d k<sub>ey</sub> d<sub>r</sub>i<sub>vers o</sub>f h<sub>e</sub>t<sub>erogeneous mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>ormance v</sub>i<sub>a</sub> SHAP (SHa<sub>p</sub>le<sub>y</sub> Additive exPlanations) anal<sub>y</sub>sis.

W<sub>e es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>p o</sub>f l<sub>oca</sub>li<sub>ze</sub>d AUC f<sub>unc</sub>ti<sub>ona</sub>l LAUC t<sub>o</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l AUC<sub>, s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t if <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> i<sub>s</sub> fl<sub>a</sub>t<sub>,</sub> th<sub>en</sub> it<sub>s</sub> <sub>average</sub> i<sub>s</sub> <sub>a</sub>l<sub>ge</sub>b<sub>ra</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>equa</sub>l t<sub>o</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l AUC<sub>.</sub> W<sub>e</sub> <sub>cons</sub>t<sub>ruc</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub> <sub>o</sub>f LAUC <sub>an</sub>d <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h it<sub>s</sub> <sub>asymp</sub>t<sub>o</sub>ti<sub>c</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>an</sub>d i<sub>n</sub>f<sub>erence.</sub> W<sub>e</sub> f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub> <sub>con</sub>diti<sub>ons</sub> th<sub>a</sub>t <sub>guaran</sub>t<sub>ee</sub> th<sub>a</sub>t LAUC i<sub>s</sub> <sub>appropr</sub>i<sub>a</sub>t<sub>e</sub>l<sub>y</sub> <sub>or</sub>d<sub>ere</sub>d <sub>an</sub>d d<sub>er</sub>i<sub>ve</sub> it<sub>s asymp</sub>t<sub>o</sub>ti<sub>c</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on an</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>error es</sub>ti<sub>ma</sub>t<sub>or.</sub> W<sub>e s</sub>h<sub>ow</sub> h<sub>ow</sub> l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> id<sub>en</sub>tifi<sub>es</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> i<sub>n c</sub>l<sub>ass</sub>ifi<sub>er</sub> di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on.</sub> W<sub>e con</sub>d<sub>uc</sub>t <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons</sub> f<sub>or</sub> b<sub>o</sub>th <sub>pos</sub>iti<sub>ve an</sub>d <sub>nega</sub>ti<sub>ve con</sub>t<sub>ro</sub>l <sub>scenar</sub>i<sub>os.</sub>

T<sub>o</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>e</sub> i<sub>n</sub>f<sub>erence an</sub>d <sub>reme</sub>di<sub>a</sub>ti<sub>on o</sub>f l<sub>ower per</sub>f<sub>ormance o</sub>f th<sub>e re</sub>f<sub>erence r</sub>i<sub>s</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>on mo</sub>d<sub>e</sub>l i<sub>n</sub> th<sub>e</sub> l<sub>ow</sub> <sub>compe</sub>t<sub>ence reg</sub>i<sub>on, we prov</sub>id<sub>e</sub> t<sub>es</sub>ti<sub>ng proce</sub>d<sub>ures.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t<sub>,</sub> i<sub>n a</sub>dditi<sub>on</sub> t<sub>o mo</sub>d<sub>e</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y,</sub> l<sub>ower</sub> <sub>per</sub>f<sub>ormance</sub> i<sub>n</sub> th<sub>e</sub> l<sub>ow compe</sub>t<sub>ence reg</sub>i<sub>on may</sub> b<sub>e</sub> d<sub>r</sub>i<sub>ven</sub> b<sub>y</sub> th<sub>e s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs o</sub>f th<sub>e va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>co</sub>h<sub>or</sub>t<sub>.</sub>

A<sub>s a prac</sub>ti<sub>ca</sub>l <sub>app</sub>li<sub>ca</sub>ti<sub>on, we use</sub>d LOCATOR t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow per</sub>f<sub>ormance reg</sub>i<sub>ons o</sub>f th<sub>e</sub> PREVENT model in the National Health and Nutrition Examination Surve<sub>y</sub> data (NHANES) 1999–2010 c<sub>y</sub>cles (N = 17,748), identified hi<sub>g</sub>h/low com<sub>p</sub>etence re<sub>g</sub>ions usin<sub>g</sub> our method, and validated hi<sub>g</sub>h/low com<sub>p</sub>etence re<sub>g</sub>ions in the UK Biobank (N=355,122), an external cohort.

## Results

Th<sub>e</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> id<sub>en</sub>tifi<sub>es</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow per</sub>f<sub>ormance reg</sub>i<sub>ons</sub> i<sub>n</sub> th<sub>e covar</sub>i<sub>a</sub>t<sub>e space.</sub> SHAP <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> id<sub>en</sub>tifi<sub>e</sub>d <sub>e</sub>d<sub>uca</sub>ti<sub>on,</sub> <sub>age,</sub> <sub>sex,</sub> <sub>curren</sub>t <sub>smo</sub>ki<sub>ng,</sub> <sub>an</sub>d <sub>preva</sub>l<sub>en</sub>t h<sub>yper</sub>t<sub>ens</sub>i<sub>on</sub> <sub>as</sub> th<sub>e</sub> <sub>mos</sub>t i<sub>mpor</sub>t<sub>an</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>nan</sub>t<sub>s</sub> <sub>o</sub>f AUC <sub>var</sub>i<sub>a</sub>ti<sub>on.</sub> L<sub>ow an</sub>d hi<sub>g</sub>h <sub>per</sub>f<sub>ormance su</sub>b<sub>groups were</sub> id<sub>en</sub>tifi<sub>e</sub>d i<sub>n</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng su</sub>b<sub>se</sub>t <sub>an</sub>d <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> h<sub>o</sub>ld <sub>ou</sub>t t<sub>es</sub>t <sub>su</sub>b<sub>se</sub>t <sub>o</sub>f NHANES<sub>.</sub> P<sub>er</sub>f<sub>ormance</sub> i<sub>n</sub> l<sub>ow su</sub>b<sub>group range</sub>d f<sub>rom .</sub>6 t<sub>o .</sub>75 <sub>an</sub>d f<sub>rom .</sub>82 t<sub>o .</sub>9 i<sub>n</sub> hi<sub>g</sub>h <sub>per</sub>f<sub>ormance su</sub>b<sub>group .</sub> Wh<sub>en sp</sub>litti<sub>ng a</sub>t th<sub>e me</sub>di<sub>an</sub> LAUC<sub>, conven</sub>ti<sub>ona</sub>l AUC <sub>va</sub>l<sub>ues</sub> i<sub>n</sub> th<sub>e</sub> l<sub>ow per</sub>f<sub>ormance</sub> sub<sub>g</sub>rou<sub>p</sub> were .78 [.69; .77] and .88 [.82; .93] in the hi<sub>g</sub>h <sub>p</sub>erformance sub<sub>g</sub>rou<sub>p</sub> in the hold-out subset of NHANES<sub>.</sub> In UK Bi<sub>o</sub>b<sub>a</sub>nk<sub>, a</sub>n <sub>ex</sub>t<sub>e</sub>rn<sub>a</sub>l <sub>co</sub>h<sub>o</sub>rt<sub>,</sub> AUC <sub>o</sub>f PREVENT m<sub>o</sub>d<sub>e</sub>l in th<sub>ese</sub> r<sub>eg</sub>i<sub>o</sub>n<sub>s p</sub>r<sub>o</sub>d<sub>uce</sub> AUC <sub>o</sub>f <sub>.</sub>76 [.74; .78] and .71 [.70; .72].

## Conclusions

Th<sub>e</sub> LOCATOR <sub>approac</sub>h <sub>o</sub>f<sub>ers a</sub> d<sub>a</sub>t<sub>a-</sub>d<sub>r</sub>i<sub>ven s</sub>t<sub>ra</sub>t<sub>egy</sub> t<sub>o</sub> id<sub>en</sub>tif<sub>y su</sub>b<sub>groups w</sub>ith hi<sub>g</sub>h <sub>an</sub>d l<sub>ow per</sub>f<sub>ormance o</sub>f th<sub>e ex</sub>i<sub>s</sub>ti<sub>ng</sub> bi<sub>nary c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on mo</sub>d<sub>e</sub>l <sub>w</sub>ith<sub>ou</sub>t l<sub>oss o</sub>f <sub>power.</sub> I<sub>n</sub> th<sub>e</sub> l<sub>ow per</sub>f<sub>ormance su</sub>b<sub>group, reme</sub>di<sub>a</sub>ti<sub>on</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>re</sub>fitti<sub>ng,</sub> <sub>enr</sub>i<sub>c</sub>hi<sub>ng</sub> th<sub>e</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> <sub>space</sub> <sub>w</sub>ith <sub>new</sub> bi<sub>omar</sub>k<sub>ers,</sub> <sub>an</sub>d id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on</sub> <sub>an</sub>d d<sub>own</sub> <sub>we</sub>i<sub>g</sub>hti<sub>ng o</sub>f <sub>reg</sub>i<sub>ons w</sub>ith l<sub>ow var</sub>i<sub>ance o</sub>f <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d PREVENT <sub>r</sub>i<sub>s</sub>k<sub>.</sub> Th<sub>e propose</sub>d l<sub>oca</sub>li<sub>ze</sub>d AUC f<sub>unc</sub>ti<sub>ona</sub>l<sub>,</sub> LAUC, ofers an objective metric that can be reported for participants in the high or low performance subgroups <sub>as measure o</sub>f th<sub>e s</sub>t<sub>reng</sub>th <sub>o</sub>f th<sub>e re</sub>f<sub>erence mo</sub>d<sub>e</sub>l<sub>.</sub>

Keywords: Model discrimination, AUC, AUROC, c-statistics, subgroup identification, binary classification model, l<sub>og</sub>i<sub>s</sub>ti<sub>c regress</sub>i<sub>on, mo</sub>d<sub>e</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y.</sub>

## 1 Introduction

Bi<sub>nary c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s are some o</sub>f th<sub>e mos</sub>t <sub>w</sub>id<sub>e</sub>l<sub>y use</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> Th<sub>e</sub>i<sub>r app</sub>li<sub>ca</sub>ti<sub>ons range</sub> f<sub>rom</sub> d<sub>e</sub>t<sub>ec</sub>ti<sub>on</sub>/<sub>-</sub> di<sub>agnos</sub>i<sub>s</sub> t<sub>o prospec</sub>ti<sub>ve r</sub>i<sub>s</sub>k <sub>s</sub>t<sub>ra</sub>tifi<sub>ca</sub>ti<sub>on.</sub> Th<sub>ere</sub>f<sub>ore, an accura</sub>t<sub>e an</sub>d <sub>nuance</sub>d <sub>assessmen</sub>t <sub>o</sub>f th<sub>e</sub>i<sub>r qua</sub>lit<sub>y o</sub>f <sub>p</sub>erformance is of <sub>p</sub>aramount im<sub>p</sub>ortance. Yet, the Area Under the Receiver O<sub>p</sub>eratin<sub>g</sub> Characteristics Curve [1] (AUROC, also known as AUC, c-index, concordance index, Harrell or Uno c-index), a widel<sub>y</sub> used measure of model <sub>p</sub>erformance, rarel<sub>y</sub> exceeds 0.80 across <sub>g</sub>uidelines-recommended models (Khan et al., Circulation 2024 [2]). It is well below its theoretical maximum of 1.0 and has room to <sub>g</sub>row. However, current methods cannot id<sub>en</sub>tif<sub>y,</sub> <sub>w</sub>ith<sub>ou</sub>t l<sub>oss</sub> <sub>o</sub>f <sub>power,</sub> <sub>w</sub>h<sub>ere</sub> thi<sub>s</sub> <sub>room</sub> i<sub>s.</sub> Th<sub>e</sub> f<sub>ocus</sub> <sub>o</sub>f thi<sub>s</sub> <sub>paper</sub> i<sub>s</sub> t<sub>o</sub> d<sub>eve</sub>l<sub>op</sub> <sub>repro</sub>d<sub>uc</sub>ibl<sub>e</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> th<sub>a</sub>t id<sub>en</sub>tif<sub>y,</sub> <sub>w</sub>ith<sub>ou</sub>t l<sub>oss</sub> <sub>o</sub>f <sub>power,</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> d<sub>ep</sub>l<sub>oye</sub>d <sub>c</sub>l<sub>ass</sub>ifi<sub>er</sub> <sub>per</sub>f<sub>orms</sub> <sub>we</sub>ll <sub>or</sub> <sub>no</sub>t <sub>so</sub> <sub>we</sub>ll<sub>,</sub> <sub>an</sub>d <sub>separa</sub>t<sub>e</sub> <sub>genu</sub>i<sub>ne</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> f<sub>rom</sub> <sub>co</sub>h<sub>or</sub>t <sub>ar</sub>tif<sub>ac</sub>t<sub>s</sub> t<sub>o</sub> <sub>gu</sub>id<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l i<sub>mprovemen</sub>t i<sub>n</sub> l<sub>ow-compe</sub>t<sub>ence</sub> <sub>reg</sub>i<sub>ons.</sub>

S<sub>evera</sub>l t<sub>rea</sub>t<sub>men</sub>t <sub>gu</sub>id<sub>e</sub>li<sub>nes</sub> <sub>re</sub>l<sub>y</sub> <sub>on</sub> <sub>o</sub>fi<sub>c</sub>i<sub>a</sub>l bi<sub>nary</sub> <sub>c</sub>l<sub>ass</sub>ifi<sub>ers</sub> th<sub>a</sub>t i<sub>n</sub>f<sub>orm</sub> <sub>r</sub>i<sub>s</sub>k <sub>s</sub>t<sub>ra</sub>tifi<sub>ca</sub>ti<sub>on</sub> f<sub>or</sub> <sub>m</sub>illi<sub>ons</sub> <sub>o</sub>f <sub>peop</sub>l<sub>e.</sub> S<sub>y</sub>stematic COronar<sub>y</sub> Risk Evaluation 2 (SCORE2) [3] and Predictin<sub>g</sub> Risk of Cardiovascular Disease EVENTs (PREVENT) [4] models are embedded in Euro<sub>p</sub>ean and US <sub>p</sub>revention <sub>g</sub>uidelines, res<sub>p</sub>ectivel<sub>y</sub>, and su<sub>pp</sub>l<sub>y</sub> the risk estimates that decide li<sub>p</sub>id- and blood-<sub>p</sub>ressure-lowerin<sub>g</sub> thera<sub>py</sub> for hundreds of millions of adults [4]. Before <sub>re</sub>l<sub>ease,</sub> th<sub>ese mo</sub>d<sub>e</sub>l<sub>s are va</sub>lid<sub>a</sub>t<sub>e</sub>d i<sub>n ex</sub>t<sub>erna</sub>l <sub>co</sub>h<sub>or</sub>t<sub>s</sub> b <sub>re or</sub>ti<sub>n</sub> di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on, ca</sub>lib<sub>ra</sub>ti<sub>on, an</sub>d <sub>rec</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on</sub> <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs.</sub> AUROC<sub>,</sub> <sub>a</sub> <sub>measure</sub> <sub>o</sub>f <sub>a</sub> <sub>mo</sub>d<sub>e</sub>l’<sub>s</sub> <sub>a</sub>bilit<sub>y</sub> t<sub>o</sub> <sub>separa</sub>t<sub>e</sub> <sub>even</sub>t<sub>s</sub> f<sub>rom</sub> <sub>non-even</sub>t<sub>s,</sub> i<sub>s</sub> <sub>a</sub> <sub>u</sub>bi<sub>qu</sub>it<sub>ous</sub> <sub>measure</sub> <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on.</sub> It i<sub>s</sub> <sub>o</sub>ft<sub>en</sub> <sub>repor</sub>t<sub>e</sub>d <sub>as</sub> <sub>one</sub> <sub>num</sub>b<sub>er</sub> <sub>per</sub> <sub>ex</sub>t<sub>erna</sub>l <sub>co</sub>h<sub>or</sub>t<sub>.</sub> H<sub>owever,</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>per-co</sub>h<sub>or</sub>t <sub>num</sub>b<sub>er</sub> <sub>canno</sub>t <sub>ru</sub>l<sub>e</sub> <sub>ou</sub>t th<sub>e</sub> <sub>poss</sub>ibilit<sub>y</sub> th<sub>a</sub>t th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>may</sub> <sub>separa</sub>t<sub>e</sub> <sub>even</sub>t<sub>s</sub> f<sub>rom</sub> <sub>non-even</sub>t<sub>s</sub> <sub>we</sub>ll i<sub>n</sub> <sub>one</sub> <sub>reg</sub>i<sub>on</sub> <sub>o</sub>f <sub>covar</sub>i<sub>a</sub>t<sub>e space an</sub>d <sub>no</sub> b<sub>e</sub>tt<sub>er</sub> th<sub>an c</sub>h<sub>ance</sub> i<sub>n ano</sub>th<sub>er.</sub> P<sub>a</sub>ti<sub>en</sub>t<sub>s</sub> i<sub>n</sub> th<sub>e wea</sub>k <sub>reg</sub>i<sub>ons rece</sub>i<sub>ve sys</sub>t<sub>ema</sub>ti<sub>ca</sub>ll<sub>y</sub> l<sub>ess</sub> <sub>accura</sub>t<sub>e</sub> <sub>r</sub>i<sub>s</sub>k <sub>es</sub>ti<sub>ma</sub>t<sub>es</sub> <sub>an</sub>d<sub>,</sub> th<sub>ere</sub>f<sub>ore,</sub> l<sub>ess</sub> <sub>appropr</sub>i<sub>a</sub>t<sub>e</sub> t<sub>rea</sub>t<sub>men</sub>t<sub>.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> <sub>accumu</sub>l<sub>a</sub>ti<sub>ng</sub> <sub>ev</sub>id<sub>ence</sub> <sub>sugges</sub>t<sub>s</sub> th<sub>a</sub>t the <sub>p</sub>erformance of risk <sub>p</sub>rediction models, includin<sub>g</sub> PREVENT, is not uniform across all <sub>p</sub>atient sub<sub>g</sub>rou<sub>p</sub>s [5] T<sub>ra</sub>diti<sub>ona</sub>l <sub>approac</sub>h<sub>es</sub> t<sub>o</sub> i<sub>nves</sub>ti<sub>ga</sub>ti<sub>ng</sub> th<sub>ese</sub> <sub>per</sub>f<sub>ormance</sub> <sub>var</sub>i<sub>a</sub>ti<sub>ons</sub> t<sub>yp</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>re</sub>l<sub>y</sub> <sub>on</sub> <sub>pre</sub>d<sub>e</sub>fi<sub>ne</sub>d<sub>,</sub> di<sub>c</sub>h<sub>o</sub>t<sub>omous</sub> sub<sub>g</sub>rou<sub>p</sub> classifications (e.<sub>g</sub>., smokers vs. non-smokers) [6]. While informative, such methods rel<sub>y</sub> on subsettin<sub>g</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a</sub> <sub>an</sub>d <sub>may</sub> f<sub>a</sub>il t<sub>o</sub> <sub>cap</sub>t<sub>ure</sub> <sub>more</sub> <sub>comp</sub>l<sub>ex,</sub> <sub>mu</sub>ltidi<sub>mens</sub>i<sub>ona</sub>l <sub>pa</sub>tt<sub>erns</sub> <sub>o</sub>f <sub>p</sub>h<sub>eno</sub>t<sub>yp</sub>i<sub>c</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> d<sub>ue</sub> t<sub>o</sub> <sub>qu</sub>i<sub>c</sub>k l<sub>oss o</sub>f <sub>power.</sub> Th<sub>ere</sub> i<sub>s no es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>es</sub>ti<sub>man</sub>d f<sub>or</sub> l<sub>oca</sub>l di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on w</sub>ith<sub>ou</sub>t <sub>su</sub>b<sub>se</sub>tti<sub>ng</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a, so</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> i<sub>s</sub> difi<sub>cu</sub>lt t<sub>o</sub> l<sub>oca</sub>t<sub>e,</sub> <sub>quan</sub>tif<sub>y,</sub> <sub>or</sub> t<sub>es</sub>t <sub>w</sub>ith<sub>ou</sub>t <sub>qu</sub>i<sub>c</sub>kl<sub>y</sub> l<sub>oos</sub>i<sub>ng</sub> <sub>power.</sub>

In <sub>p</sub>arallel, the conce<sub>p</sub>t of <sub>p</sub>henoma<sub>pp</sub>in<sub>g</sub> has recentl<sub>y</sub> emer<sub>g</sub>ed [7]. Develo<sub>p</sub>ed to assess hetero<sub>g</sub>eneit<sub>y</sub> of treatment <sub>e</sub>f<sub>ec</sub>t i<sub>n ran</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>con</sub>t<sub>ro</sub>ll<sub>e</sub>d t<sub>r</sub>i<sub>a</sub>l<sub>s,</sub> it l<sub>everages</sub> th<sub>e</sub> f<sub>u</sub>ll b<sub>rea</sub>dth <sub>o</sub>f d<sub>a</sub>t<sub>a.</sub> B<sub>y quan</sub>tif<sub>y</sub>i<sub>ng</sub> i<sub>n</sub>t<sub>er-</sub>i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l similarit<sub>y</sub> based on multi<sub>p</sub>le clinical features (e.<sub>g</sub>., usin<sub>g</sub> Gower distance [8]), <sub>p</sub>henoma<sub>pp</sub>in<sub>g</sub> allows for the di<sub>scovery</sub> <sub>o</sub>f <sub>con</sub>ti<sub>guous,</sub> d<sub>a</sub>t<sub>a-</sub>d<sub>r</sub>i<sub>ven</sub> <sub>pa</sub>ti<sub>en</sub>t <sub>su</sub>b<sub>groups</sub> <sub>w</sub>ith h<sub>e</sub>t<sub>erogeneous</sub> t<sub>rea</sub>t<sub>men</sub>t <sub>e</sub>f<sub>ec</sub>t<sub>.</sub> Thi<sub>s</sub> <sub>approac</sub>h h<sub>as</sub> been successfull<sub>y</sub> a<sub>pp</sub>lied to identif<sub>y</sub> hetero<sub>g</sub>eneit<sub>y</sub> of treatment efects in clinical trials [7]. It relies on estimatin<sub>g</sub> th<sub>e average per-person</sub> h<sub>azar</sub>d<sub>, an</sub>d th<sub>ere</sub>f<sub>ore canno</sub>t b<sub>e</sub> di<sub>rec</sub>tl<sub>y</sub> t<sub>rans</sub>l<sub>a</sub>t<sub>e</sub>d t<sub>o</sub> AUROC<sub>.</sub> I<sub>n</sub> thi<sub>s paper, we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>a</sub> l<sub>oca</sub>li<sub>ze</sub>d AUC f<sub>unc</sub>ti<sub>ona</sub>l LAUC<sub>,</sub> th<sub>a</sub>t<sub>,</sub> f<sub>or a re</sub>f<sub>erence r</sub>i<sub>s</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>, es</sub>ti<sub>ma</sub>t<sub>es</sub> AUC f<sub>or an</sub> i<sub>n</sub>d<sub>ex</sub> <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t i <sub>us</sub>i<sub>ng a</sub>ll <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s we</sub>i<sub>g</sub>ht<sub>e</sub>d b<sub>y s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> t<sub>o</sub> i<sub>.</sub> B<sub>y us</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d AUC f<sub>unc</sub>ti<sub>ona</sub>l<sub>, our</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>avo</sub>id<sub>s</sub> <sub>su</sub>b<sub>se</sub>tti<sub>ng</sub> d<sub>a</sub>t<sub>a</sub> <sub>a</sub>lt<sub>oge</sub>th<sub>er.</sub>

Then we use localized AUC functional (LAUC) to identif<sub>y</sub> conti<sub>g</sub>uous <sub>p</sub>henot<sub>yp</sub>ic sub<sub>g</sub>rou<sub>p</sub>s in which the reference bi<sub>nary c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on mo</sub>d<sub>e</sub>l <sub>ex</sub>hibit<sub>s su</sub>b<sub>op</sub>ti<sub>ma</sub>l di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on.</sub> W<sub>e</sub> f<sub>ur</sub>th<sub>er see</sub>k t<sub>o c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> th<sub>e</sub> f<sub>ea</sub>t<sub>ures</sub> th<sub>a</sub> d<sub>r</sub>i<sub>ve</sub> thi<sub>s</sub> <sub>per</sub>f<sub>ormance</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y.</sub>

W<sub>e</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>p</sub> <sub>o</sub>f l<sub>oca</sub>li<sub>ze</sub>d AUC f<sub>unc</sub>ti<sub>ona</sub>l LAUC t<sub>o</sub> th<sub>e</sub> <sub>conven</sub>ti<sub>ona</sub>l <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l AUC<sub>.</sub> W<sub>e</sub> <sub>prov</sub>id<sub>e</sub> t<sub>wo</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>ons</sub> <sub>an</sub>d <sub>prove</sub> th<sub>a</sub>t th<sub>ey</sub> <sub>guaran</sub>t<sub>y</sub> th<sub>e</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>w</sub>ill id<sub>en</sub>tif<sub>y</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow</sub> <sub>per</sub>f<sub>ormance</sub> <sub>reg</sub>i<sub>ons</sub> i<sub>n</sub> th<sub>e</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> <sub>space.</sub> W<sub>e</sub> <sub>cons</sub>t<sub>ruc</sub>t <sub>an</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub> <sub>o</sub>f LAUC <sub>an</sub>d <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h it<sub>s</sub> <sub>asymp</sub>t<sub>o</sub>ti<sub>c</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>an</sub>d <sub>a</sub> <sub>c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>or</sub> f<sub>or</sub> th<sub>e</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>error.</sub>

T<sub>o</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>e</sub> i<sub>n</sub>f<sub>erence an</sub>d <sub>reme</sub>di<sub>a</sub>ti<sub>on o</sub>f l<sub>ower per</sub>f<sub>ormance o</sub>f th<sub>e re</sub>f<sub>erence r</sub>i<sub>s</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>on mo</sub>d<sub>e</sub>l i<sub>n</sub> th<sub>e</sub> l<sub>ow</sub> LOCATOR <sub>reg</sub>i<sub>on,</sub> <sub>we</sub> <sub>prov</sub>id<sub>e</sub> t<sub>es</sub>ti<sub>ng</sub> <sub>proce</sub>d<sub>ures.</sub> W<sub>e</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t<sub>,</sub> i<sub>n</sub> <sub>a</sub>dditi<sub>on</sub> t<sub>o</sub> <sub>mo</sub>d<sub>e</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y,</sub> l<sub>ower</sub> <sub>per</sub>f<sub>ormance</sub> i<sub>n</sub> th<sub>e</sub> l<sub>ow</sub> <sub>compe</sub>t<sub>ence</sub> <sub>reg</sub>i<sub>on</sub> <sub>may</sub> b<sub>e</sub> d<sub>r</sub>i<sub>ven</sub> b<sub>y</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub> <sub>o</sub>f th<sub>e</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>co</sub>h<sub>or</sub>t<sub>.</sub>

O<sub>ur</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>prov</sub>id<sub>es</sub> <sub>a</sub> <sub>nove</sub>l<sub>,</sub> d<sub>a</sub>t<sub>a-</sub>d<sub>r</sub>i<sub>ven</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> f<sub>or</sub> t<sub>arge</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>re</sub>fi<sub>nemen</sub>t <sub>w</sub>hil<sub>e</sub> <sub>avo</sub>idi<sub>ng</sub> l<sub>oss</sub> <sub>o</sub>f <sub>power.</sub> It <sub>a</sub>l<sub>so</sub> <sub>prov</sub>id<sub>es</sub> <sub>a</sub> <sub>more</sub> <sub>nuance</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> f<sub>or</sub> <sub>mo</sub>d<sub>e</sub>l <sub>re</sub>fi<sub>nemen</sub>t<sub>,</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t it i<sub>s</sub> <sub>an</sub> i<sub>n</sub>t<sub>erp</sub>l<sub>ay</sub> <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y,</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>co</sub>h<sub>or</sub>t <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs,</sub> <sub>an</sub>d <sub>over</sub>l<sub>ap</sub> i<sub>n</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>r</sub>i<sub>s</sub>k i<sub>n</sub> <sub>cer</sub>t<sub>a</sub>i<sub>n</sub> <sub>reg</sub>i<sub>ons</sub> <sub>o</sub>f <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> <sub>space.</sub>

## 2 Properties of The Localized AUC and Asymptotic Theory

## 2.1 Definition of Localized AUC Functional

Th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d AUC i<sub>s</sub> <sub>a</sub> <sub>mo</sub>difi<sub>e</sub>d <sub>vers</sub>i<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>we</sub>ll<sub>-</sub>k<sub>nown</sub> AUC<sub>.</sub> It<sub>s</sub> <sub>ma</sub>i<sub>n</sub> <sub>purpose</sub> i<sub>s</sub> t<sub>o</sub> <sub>g</sub>i<sub>ve</sub> <sub>a</sub> <sub>persona</sub>li<sub>ze</sub>d <sub>per</sub>f<sub>ormance</sub> <sub>o</sub>f th<sub>e</sub> <sub>r</sub>i<sub>s</sub>k <sub>pre</sub>di<sub>c</sub>t<sub>or.</sub>

Definition 1 (Localized AUC Functional). Let X denote covariate space. Ifwe sample a patient with an event and covariates $X ^ { + } \sim F ^ { + }$ , where $F ^ { + }$ is the sampling distribution of our event patients, and a patient with no event with covariates $X ^ { - } \sim F ^ { - }$ . Let $x _ { i } \in \mathcal X$ be covariates for our index patient i . We also have a risk prediction model $f : \mathcal { X }  [ 0 , 1 ]$ . In addition, we also have a weight function $w : \mathcal { X } \times \mathcal { X }  [ 0 , 1 ]$ , that describes the similarity between patients. Then the localized AUC is as follows:

$$
\begin{array} { r } { \mathrm { L A U C } ( x _ { i } ) = \frac { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( x _ { i } , X ^ { + } ) w ( x _ { i } , X ^ { - } ) \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) \right] } { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( x _ { i } , X ^ { + } ) w ( x _ { i } , X ^ { - } ) \right] } } \end{array}\tag{1}
$$

Definition 2 (Localized AUC Estimator). Let’s say our dataset includes K patients with events with covariates $x _ { 1 } ^ { + } , x _ { 2 } ^ { + } , \ldots , x _ { K } ^ { + }$ and we also have L patients that have no events with covariates $x _ { 1 } ^ { - } , x _ { 2 } ^ { - } , \ldots , x _ { L } ^ { - }$ . We also have a risk prediction model $f : \mathcal { X } \to [ 0 , 1 ]$ . In addition, we also have a weightfunction $v : \mathcal { X } \times \mathcal { X }  [ 0 , 1 ]$ . Then the localized AUC estimator for some sample patient $x _ { i } \in \mathcal X$ is

$$
\widehat { \mathrm { L A U C } } ( x _ { i } ) = \frac { \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } w ( x _ { i } , x _ { k } ^ { + } ) w ( x _ { i } , x _ { l } ^ { - } ) \mathbb { 1 } \left( f ( x _ { k } ^ { + } ) > f ( x _ { l } ^ { - } ) \right) } { \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } w ( x _ { i } , x _ { k } ^ { + } ) w ( x _ { i } , x _ { l } ^ { - } ) }\tag{2}
$$

I<sub>n</sub> <sub>our</sub> <sub>paper</sub> th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht f<sub>unc</sub>ti<sub>on</sub> <sub>w</sub>ill b<sub>e</sub> <sub>a</sub> k<sub>erne</sub>l<sub>,</sub> th<sub>a</sub>t i<sub>s</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> G<sub>ower</sub> di<sub>s</sub>t<sub>ance.</sub>

Definition 3 (Gower distance [8]). Let $x , y \in { \mathcal { X } }$ , where X is a P dimensional covariate space. Suppose that $\forall p = 1 , \ldots , P ,$ , the ranges $o f x _ { p }$ and $y _ { p }$ are both $R ( p )$ . Then the Gower distance is

$$
d ( x , y ) = \sum _ { p = 1 } ^ { P } 1 - \frac { | x _ { p } - y _ { p } | } { R ( p ) } .\tag{3}
$$

I<sub>n</sub> <sub>our</sub> <sub>paper,</sub> th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht f<sub>unc</sub>ti<sub>on</sub> <sub>w</sub>ill b<sub>e</sub> $w ( x , y ) = ( 1 - d ( x , y ) ) ^ { 1 5 }$

## 2.2 Theoretical Properties of Localized AUC Functional

## 2.2.1 Relationship of $\mathrm { L A U C } ( x _ { i } )$ to global AUC under uniform similarity

If th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>s</sub> <sub>are</sub> <sub>un</sub>if<sub>orm,</sub> <sub>we</sub> <sub>en</sub>d <sub>up</sub> <sub>w</sub>ith th<sub>e</sub> t<sub>ra</sub>diti<sub>ona</sub>l AUC<sub>.</sub>

Proposition 1 (Under Uniform Weights the Localized AUC Reduces to the AUC). Let X denote the covariate space. If we sample a patient with an event and covariates $X ^ { + } \sim F ^ { + }$ , where $F ^ { + }$ is the sampling distribution of our event patients, and a patient with no event with covariates $X ^ { - } \sim F ^ { - }$ . Let $x _ { i } \in { \mathcal { X } }$ be covariates for our index patient i. We also have a risk prediction model f $: \mathcal { X }  [ 0 , 1 ]$ . In addition, we also have a weightfunction $w : \mathcal { X } \times \mathcal { X }  [ 0 , 1 ]$ , that is some constant c. Then the localized AUC is equivalent to the traditional AUC:

$$
\mathrm { L A U C } ( x _ { i } ) = \mathrm { A U C } .\tag{4}
$$

Th<sub>e</sub> <sub>proo</sub>f <sub>can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> th<sub>e</sub> S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>ma</sub>t<sub>er</sub>i<sub>a</sub>l 9

## 2.2.2 Localized AUC as Covariance

Th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d AUC h<sub>as</sub> th<sub>e</sub> <sub>proper</sub>t<sub>y</sub> <sub>as</sub> P<sub>ropos</sub>iti<sub>on</sub> 2<sub>:</sub>

Proposition 2 (Localized AUC as Covariance). Suppose number of events is n and number of non-events is m. Define $W _ { k l } = w _ { k } ^ { + } w _ { l } ^ { - }$ where $w ^ { + }$ is the weight vector of event participants, and $w ^ { - }$ is the weight vector of non-event participants. Then $W \in R ^ { n \times m }$ With two types ofweight vectors $w _ { i } , w _ { j }$ , The $W ^ { ( i ) } , W ^ { ( j ) }$ can be calculated. For event participants $k , a$ and non-event participants $l , b ,$ define $\begin{array} { r } { I _ { k l } = \mathbb { 1 } \big ( f \big ( x _ { k } \big ) > f \big ( x _ { l } \big ) \big ) + \frac { 1 } { 2 } \mathbb { 1 } \big ( f \big ( x _ { k } \big ) = f \big ( x _ { l } \big ) \big ) \in R ^ { n \times m } } \end{array}$ $\begin{array} { r } { \gamma _ { k l } = \frac { W _ { k l } ^ { ( i ) } } { W _ { k l } ^ { ( j ) } } } \end{array}$ and $\begin{array} { r } { p _ { k l } = \frac { W _ { k l } ^ { ( j ) } } { \sum _ { a , b } W _ { a b } ^ { ( j ) } } } \end{array}$ . If we think of p as a probability distribution of k, l, then $\mathrm { L A U C } ( x _ { i } ) - \mathrm { L A U C } ( x _ { j } ) =$ $\frac { C o \nu _ { p } ( \gamma , I ) } { \mathbb { E } _ { 1 } ( \gamma ) }$

Th<sub>e</sub> <sub>proo</sub>f <sub>can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>ma</sub>t<sub>er</sub>i<sub>a</sub>l 9<sub>.</sub>

## 2.2.3 Monotonicity Guarantees

L<sub>e</sub>t th<sub>e</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> <sub>space</sub> $\mathcal { X }$ be partitioned into two disjoint, phenotypically distinct regions $S _ { 1 }$ <sub>an</sub>d $S _ { 2 }$ <sub>.</sub> We a<sub>ssu</sub>me $S _ { 1 } \cap S _ { 2 } = \emptyset$ <sub>an</sub>d $S _ { 1 } \cup S _ { 2 } = \mathcal { X }$ <sub>.</sub> Th<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d AUC <sub>score</sub> <sub>can</sub> b<sub>e</sub> <sub>regar</sub>d<sub>e</sub>d <sub>as</sub> <sub>a</sub> <sub>per</sub>f<sub>ormance</sub> <sub>score</sub> <sub>o</sub>f <sub>a</sub> <sub>spec</sub>ifi<sub>c</sub> model in participants phenot<sub>y</sub>picall<sub>y</sub> similar to i. We claim using a simplified scenario with some assumptions:

Proposition 3 (Ordering of Localized AUC Expectations Across Model Competence Regions). IfAssumptions 1 and 2 hold, then the population regional expectations of localized AUC functionals satisfy:

$$
\mathbb { E } _ { X _ { i } \sim P _ { X | S _ { 1 } } } [ \mathrm { L A U C } ( X _ { i } ) ] > \mathbb { E } _ { X _ { j } \sim P _ { X | S _ { 2 } } } [ \mathrm { L A U C } ( X _ { j } ) ]
$$

This justifies why the weighted AUC is a plausible metric for model performance evaluation. The assumptions <sub>are</sub> i<sub>n</sub> S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>ma</sub>t<sub>er</sub>i<sub>a</sub>l 9<sub>.</sub>1<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>proo</sub>f i<sub>s</sub> i<sub>n</sub> 9<sub>.</sub>3<sub>.</sub>

<sub>To stabilize the variance and address boundary issues</sub>∗<sub>, we applied the logit transformation:</sub>

$$
z _ { i } = \log \left( \frac { \mathrm { L A U C } ( x _ { i } ) } { 1 - \mathrm { L A U C } ( x _ { i } ) } \right) ,
$$

<sub>w</sub>h<sub>ere</sub> $z _ { i }$ <sub>was</sub> <sub>use</sub>d <sub>as</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> f<sub>or</sub> th<sub>e</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>mac</sub>hi<sub>ne</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub>

## 2.3 Estimator Construction and Asymptotic Consistency

A<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 2<sub>, we can</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e es</sub>ti<sub>ma</sub>t<sub>or o</sub>f th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d AUC<sub>.</sub>

Proposition 4. If we follow Definitions 1 and 2, then the $\widehat { \mathrm { L A U C } }$ estimator is an unbiased consistent estimator of LAUC.

D<sub>e</sub>t<sub>a</sub>il<sub>s o</sub>f thi<sub>s can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> th<sub>e</sub> S<sub>upp</sub>l<sub>emen</sub>t<sub>ary ma</sub>t<sub>er</sub>i<sub>a</sub>l 9<sub>.</sub>4<sub>.</sub>

## 2.4 Asymptotic Normality

Proposition 5 (Asymptotic Normality).

$$
\frac { 1 } { N } ( \widehat { \mathrm { L A U C } } ( x _ { i } ) - \mathrm { L A U C } ( x _ { i } ) ) \overset { d } {  } \mathcal { N } ( 0 , \frac { \sigma ^ { 2 } } { ( \mu _ { + } \mu _ { - } ) ^ { 2 } } ) , \quad w h e r e \quad \sigma ^ { 2 } = \frac { \sigma _ { + } ^ { 2 } } { \lambda } + \frac { \sigma _ { - } ^ { 2 } } { 1 - \lambda } .\tag{5}
$$

Th<sub>e</sub> <sub>proo</sub>f <sub>can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> th<sub>e</sub> S<sub>upp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>ma</sub>t<sub>er</sub>i<sub>a</sub>l 9<sub>.</sub>4<sub>.</sub>

## 2.5 Plug-in Standard Errors via Logit Transformation

Proposition 6. An asymptotically valid $1 0 0 ( 1 - \alpha ) \%$ confidence interval is as follows as $N  \infty .$

$$
\mathbb { P } ( \mathrm { L A U C } \in [ \widehat { \mathrm { L A U C } } _ { N } - z _ { 1 - \alpha / 2 } \frac { \widehat \sigma _ { \mathrm { L A U C } } } { \sqrt { N } } , \widehat { \mathrm { L A U C } } _ { N } + z _ { 1 - \alpha / 2 } \frac { \widehat \sigma _ { \mathrm { L A U C } } } { \sqrt { N } } ] )  1 - \alpha
$$

where $z _ { 1 - \alpha / 2 }$ is the standard normal critical value.

Th<sub>e</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>on o</sub>f th<sub>e con</sub>fid<sub>ence</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> 9<sub>.</sub>4

## 3 Proposed LOCATOR Algorithm for Subgroup Identification

## 3.1 Mathematical Formulation

C<sub>ons</sub>id<sub>er</sub> th<sub>e</sub> i<sub>npu</sub>t f<sub>ea</sub>t<sub>ure ma</sub>t<sub>r</sub>i<sub>x</sub> $\boldsymbol { X } \in \mathbb { R } ^ { n \times P }$ <sub>w</sub>ith $X ~ = ~ [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ]$ <sub>an</sub>d $x _ { i } \in \mathbb { R } ^ { P }$ <sub>an</sub>d <sub>an</sub> <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> bi<sub>nary</sub> classification model f (e.<sub>g</sub>. PREVENT in our case) that is bein<sub>g</sub> evaluated b<sub>y</sub> the method. The in<sub>p</sub>ut data <sub>s</sub>h<sub>ou</sub>ld i<sub>nc</sub>l<sub>u</sub>d<sub>e a</sub>ll th<sub>e pre</sub>di<sub>c</sub>t<sub>ors o</sub>f th<sub>e ex</sub>i<sub>s</sub>ti<sub>ng mo</sub>d<sub>e</sub>l b<sub>e</sub>i<sub>ng eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>an</sub>d th<sub>e var</sub>i<sub>a</sub>bl<sub>es nee</sub>d<sub>e</sub>d t<sub>o</sub> dif<sub>eren</sub>ti<sub>a</sub>t<sub>e</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance</sub> f<sub>rom</sub> l<sub>ow-per</sub>f<sub>ormance</sub> <sub>reg</sub>i<sub>ons.</sub> D<sub>eno</sub>t<sub>e</sub> $S$ <sub>as</sub> th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f <sub>pre</sub>di<sub>c</sub>t<sub>or</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> <sub>o</sub>f th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l $f ;$ th<sub>us,</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>vec</sub>t<sub>or</sub> <sub>can</sub> b<sub>e</sub> <sub>compu</sub>t<sub>e</sub>d <sub>as</sub> $\hat { p } = [ f ( x _ { 1 , S } ) , . . . , f ( x _ { n , S } ) ]$

To <sub>q</sub>uantif<sub>y p</sub>henot<sub>yp</sub>ic similarit<sub>y</sub> between individuals, we used Gower distance [8], which accommodates mixed <sub>con</sub>ti<sub>nuous</sub> <sub>an</sub>d bi<sub>nary</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es.</sub> L<sub>e</sub>t $\mathbf { x } _ { i }$ <sub>an</sub>d $\mathbf { x } _ { j }$ be the baseline feature vectors for participants i and $j .$ <sub>.</sub> Th<sub>e</sub> G<sub>owe</sub>r di<sub>s</sub>t<sub>ance</sub> $d _ { i j }$ i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> th<sub>e</sub> <sub>average</sub> <sub>o</sub>f <sub>per-var</sub>i<sub>a</sub>bl<sub>e</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons:</sub>

$$
d _ { i j } = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \delta _ { i j p } ,
$$

<sub>w</sub>h<sub>ere</sub> f<sub>or a con</sub>ti<sub>nuous var</sub>i<sub>a</sub>bl<sub>e,</sub> $\begin{array} { r } { \delta _ { i j p } = \frac { | x _ { i p } - x _ { j p } | } { \mathrm { r a n g e } _ { p } } } \end{array}$ <sub>,</sub> <sub>an</sub>d f<sub>or</sub> <sub>a</sub> bi<sub>nary</sub> <sub>var</sub>i<sub>a</sub>bl<sub>e,</sub> $\delta _ { i j p } = { \bf 1 } ( x _ { i p } \neq x _ { j p } )$ <sub>.</sub> Th<sub>e</sub> di<sub>s</sub>t<sub>ance</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $\mathbf { D } = \left( d _ { i j } \right)$ <sub>was</sub> <sub>compu</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> <sub>a</sub>ll b<sub>ase</sub>li<sub>ne</sub> <sub>p</sub>h<sub>eno</sub>t<sub>yp</sub>i<sub>c</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> $P$

For each index participant i, we define similarit<sub>y</sub> weights:

$$
w _ { i j } = \left( 1 - d _ { i j } \right) ^ { 1 5 } ,
$$

followed b<sub>y</sub> truncation to ensure non-ne<sub>g</sub>ative wei<sub>g</sub>hts, consistent with <sub>p</sub>rior work [7]. Thus, <sub>p</sub>artici<sub>p</sub>ants <sub>p</sub>henot<sub>y</sub>picall<sub>y</sub> closer to i receive higher weights, forming a local neighborhood around i.

F<sub>or eac</sub>h i<sub>n</sub>d<sub>ex par</sub>ti<sub>c</sub>i<sub>pan</sub>t $i ,$ <sub>we eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e</sub> di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on per</sub>f<sub>ormance o</sub>f th<sub>e re</sub>f<sub>erence mo</sub>d<sub>e</sub>l <sub>w</sub>ithi<sub>n</sub> it<sub>s</sub> l<sub>oca</sub>l <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d b<sub>y</sub> <sub>compu</sub>ti<sub>ng</sub> it l<sub>oca</sub>li<sub>ze</sub>d AUC <sub>-</sub> <sub>a</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>area</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>rece</sub>i<sub>ver</sub> <sub>opera</sub>ti<sub>ng</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>c</sub> <sub>curve</sub> (wei<sub>g</sub>hted AUC)[9]:

$$
\mathrm { L A U C } ( x _ { i } ) = \frac { \sum _ { k , l : y _ { k } = 1 , y _ { l } = 0 } [ \mathbf { 1 } ( \hat { p } _ { k } > \hat { p } _ { l } ) + \frac { 1 } { 2 } \mathbf { 1 } ( \hat { p } _ { k } = \hat { p } _ { l } ) ] w _ { i k } w _ { i l } } { \sum _ { k , l : y _ { k } = 1 , y _ { l } = 0 } w _ { i k } w _ { i l } } ,
$$

<sub>w</sub>h<sub>ere</sub> $y$ is the true outcome (CVD death) and $\hat { p }$ i<sub>s</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>r</sub>i<sub>s</sub>k <sub>o</sub>f th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> Thi<sub>s</sub> <sub>pro</sub>d<sub>uces</sub> <sub>an</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l<sub>-</sub>f<sub>ocuse</sub>d l<sub>oca</sub>li<sub>ze</sub>d AUC <sub>va</sub>l<sub>ue,</sub> $\mathrm { L A U C } ( x _ { i } )$ <sub>,</sub> <sub>re</sub>fl<sub>ec</sub>ti<sub>ng</sub> h<sub>ow</sub> <sub>we</sub>ll PREVENT di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>t<sub>es</sub> CVD <sub>r</sub>i<sub>s</sub>k i<sub>n</sub>

$$
i .
$$

To predict LAUC we train XGBoost regression model using logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$ <sub>as an ou</sub>t<sub>come an</sub>d <sub>us</sub>i<sub>ng</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>p</sub><sup>h</sup>enot<sub>yp</sub><sup>i</sup>c <sup>f</sup>eatures $x _ { i } \in \mathbb { R } ^ { P }$

## 3.2 LOCATOR algorithm and implementation details

Algorithm 1 LOCATOR Algorithm   
Require: Input data matrix $\ b { X } \in \mathbb { R } ^ { n \times d } ,$ , true outcomes y, existing model f predictions ${ \hat { p } } = f ( X )$   
Ensure: Identify high and low competence regions of risk prediction model ${ \hat { p } } = f ( X )$   
1<sub>:</sub> S<sub>p</sub>lit i<sub>n</sub>t<sub>o</sub> t<sub>ra</sub>i<sub>n an</sub>d t<sub>es</sub>t d<sub>a</sub>t<sub>ase</sub>t<sub>:</sub> $X _ { \mathrm { t r a i n } } , X _ { \mathrm { t e s t } } , y _ { \mathrm { t r a i n } } , y _ { \mathrm { t e s t } }  \mathrm { s p l i t } ( X , y )$ <sub>.</sub> Li<sub>ne</sub> $2 - 9$ <sub>are</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng process, an</sub>d li<sub>nes</sub>   
10 − 11 are testing process.   
2<sub>:</sub> C<sub>ompu</sub>t<sub>e</sub> G<sub>ower</sub> di<sub>s</sub>t<sub>ances us</sub>i<sub>ng a</sub>ll b<sub>ase</sub>li<sub>ne var</sub>i<sub>a</sub>bl<sub>es:</sub> $d _ { k l } \gets \mathbf { G o w e r } ( x _ { k } , x _ { l } ) , x _ { k } \in X _ { \mathrm { t r a i n } } , x _ { l } \in X _ { \mathrm { t r a i n } }$   
3<sub>:</sub> C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>s:</sub> $w _ { k l } \gets ( 1 - d _ { k l } ) ^ { 1 5 }$   
4: for Reference participant i = 1 to $| X _ { \mathrm { t r a i n } } |$ do   
5<sub>:</sub> $\begin{array} { r } { \mathrm { L A U C } ( x _ { i } ) = \frac { \sum _ { k , l : y _ { k } = 1 , y _ { l } = 0 } [ \mathbf { 1 } ( \hat { p } _ { k } > \hat { p } _ { l } ) + \frac { 1 } { 2 } \mathbf { 1 } ( \hat { p } _ { k } = \hat { p } _ { l } ) ] w _ { i k } w _ { i l } } { \sum _ { k , l : y _ { k } = 1 , y _ { l } = 0 } w _ { i k } w _ { i l } } \mathrm { L A U C } ( x _ { i } ) } \end{array}$   
6: end for   
7: Train XGBoost model to predict logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$ : gˆ<sub>XGB</sub> $: X _ { \mathrm { t r a i n } } \mapsto$ logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$   
8: Select features that are important in the final BGBoost model gˆ<sub>XGB</sub>   
9<sub>:</sub> Use $\hat { g } _ { \mathrm { X G B } }$ t<sub>o ge</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>ons</sub> i<sub>n</sub> t<sub>es</sub>t <sub>se</sub>t<sub>:</sub> $z _ { \mathrm { p r e d } }  \hat { g } _ { \mathrm { X G B } } ( X _ { \mathrm { t e s t } } )$   
10<sub>:</sub> Id<sub>en</sub>tif<sub>y</sub> l<sub>ow- an</sub>d hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>groups:</sub>   
• median = median $( z _ { \mathrm { p r e d } } )$   
$X _ { \mathrm { l o w } }  z _ { \mathrm { p r e d } } <$ <sub>me</sub>di<sub>an</sub>   
• X<sub>high</sub> ← z<sub>pred</sub> >= median   
11<sub>:</sub> R<sub>un</sub> SHAP <sub>or o</sub>th<sub>er s</sub>i<sub>m</sub>il<sub>ar ana</sub>l<sub>ys</sub>i<sub>s</sub> t<sub>o</sub> id<sub>en</sub>tif<sub>y w</sub>h<sub>a</sub>t ki<sub>n</sub>d <sub>o</sub>f <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s cons</sub>tit<sub>u</sub>t<sub>e</sub> l<sub>ow an</sub>d hi<sub>g</sub>h <sub>per</sub>f<sub>ormance</sub>   
su<sup>b</sup><sub>g</sub>rou<sub>p</sub>s <sup>i</sup>n $X _ { \mathrm { t r a i n } }$

## 3.3 Interpretation of High and Low Performance Regions with SHAP analysis

T<sub>o</sub> int<sub>e</sub>r<sub>p</sub>r<sub>e</sub>t th<sub>e</sub> XGB<sub>oos</sub>t m<sub>o</sub>d<sub>e</sub>l ${ \hat { g } } _ { \mathrm { X G B } } ( X _ { \mathrm { t e s t } } )$ <sub>an</sub>d id<sub>en</sub>tif<sub>y</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>ne c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub> d<sub>r</sub>i<sub>v</sub>i<sub>ng</sub> l<sub>oca</sub>l <sub>per</sub>f<sub>ormance,</sub> we <sub>p</sub>ro<sub>p</sub>ose to em<sub>p</sub>lo<sub>y</sub> SHa<sub>p</sub>le<sub>y</sub> Additive exPlanations (SHAP) [10]. For each test sam<sub>p</sub>le, SHAP value is com<sub>p</sub>uted f<sub>or</sub> <sub>every</sub> f<sub>ea</sub>t<sub>ure,</sub> <sub>represen</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>a</sub>t f<sub>ea</sub>t<sub>ure</sub> t<sub>o</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d $z _ { i }$ <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> <sub>average</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on.</sub> A SHAP <sub>summary</sub> <sub>p</sub>l<sub>o</sub>t <sub>can</sub> b<sub>e</sub> <sub>use</sub>d t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e</sub> <sub>or</sub>d<sub>er</sub> <sub>o</sub>f i<sub>mpor</sub>t<sub>ance</sub> <sub>o</sub>f <sub>eac</sub>h <sub>var</sub>i<sub>a</sub>bl<sub>e.</sub> Th<sub>us,</sub> <sub>su</sub>b<sub>groups</sub> <sub>can</sub> b<sub>e</sub> id<sub>en</sub>tifi<sub>e</sub>d <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub> SHAP <sub>va</sub>l<sub>ues</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> <sub>mos</sub>t i<sub>mpor</sub>t<sub>an</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>es.</sub>

W<sub>e</sub> d<sub>er</sub>i<sub>ve</sub>d <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l f<sub>ea</sub>t<sub>ure</sub> i<sub>mpor</sub>t<sub>ance</sub> <sub>as</sub> th<sub>e</sub> <sub>mean</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> SHAP <sub>va</sub>l<sub>ue</sub> <sub>per</sub> f<sub>ea</sub>t<sub>ure.</sub> Additi<sub>ona</sub>ll<sub>y,</sub> <sub>we</sub> <sub>v</sub>i<sub>sua</sub>li<sub>ze</sub>d th<sub>e</sub> di<sub>rec</sub>ti<sub>on</sub> <sub>an</sub>d <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> <sub>o</sub>f <sub>eac</sub>h f<sub>ea</sub>t<sub>ure</sub>’<sub>s</sub> <sub>e</sub>f<sub>ec</sub>t <sub>v</sub>i<sub>a</sub> <sub>summary</sub> <sub>p</sub>l<sub>o</sub>t<sub>s.</sub> F<sub>ea</sub>t<sub>ures</sub> <sub>w</sub>ith hi<sub>g</sub>h i<sub>mpor</sub>t<sub>ance</sub> <sub>were</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d f<sub>or</sub> id<sub>en</sub>tif<sub>y</sub>i<sub>ng</sub> h<sub>e</sub>t<sub>erogeneous</sub> <sub>per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group,</sub> <sub>an</sub>d th<sub>e</sub>i<sub>r</sub> <sub>e</sub>f<sub>ec</sub>t<sub>s</sub> <sub>on</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d l<sub>oca</sub>l <sub>per</sub>f<sub>ormance</sub> <sub>were</sub> f<sub>ur</sub>th<sub>er</sub> <sub>exam</sub>i<sub>ne</sub>d <sub>us</sub>i<sub>ng</sub> SHAP d<sub>epen</sub>d<sub>ence</sub> <sub>p</sub>l<sub>o</sub>t<sub>s.</sub>

T<sub>o a</sub>dd<sub>ress po</sub>t<sub>en</sub>ti<sub>a</sub>l i<sub>ns</sub>t<sub>a</sub>bilit<sub>y</sub> d<sub>ue</sub> t<sub>o over</sub>fitti<sub>ng, we a</sub>l<sub>so ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub>d th<sub>e</sub> i<sub>n</sub>t<sub>erna</sub>l i<sub>mpor</sub>t<sub>ance o</sub>f XGB<sub>oos</sub>t ’<sub>ga</sub>i<sub>n</sub> <sub>an</sub>d <sub>compare</sub>d it <sub>w</sub>ith th<sub>e</sub> SHAP <sub>resu</sub>lt<sub>s</sub> <sub>as</sub> <sub>a</sub> <sub>sens</sub>iti<sub>v</sub>it<sub>y</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>an</sub>d <sub>remove</sub>d f<sub>ea</sub>t<sub>ures</sub> <sub>w</sub>ith <sub>ga</sub>i<sub>n</sub> <sub>wea</sub>k<sub>er</sub> th<sub>an</sub> 10% <sub>o</sub>f th<sub>e</sub> <sub>mos</sub>t i<sub>mpor</sub>t<sub>an</sub>t f<sub>ea</sub>t<sub>ure..</sub>

I<sub>n</sub> <sub>a</sub>dditi<sub>on,</sub> t<sub>o</sub> <sub>ver</sub>if<sub>y</sub> th<sub>a</sub>t th<sub>e</sub> XGB<sub>oos</sub>t <sub>mo</sub>d<sub>e</sub>l h<sub>as</sub> th<sub>e</sub> <sub>a</sub>bilit<sub>y</sub> t<sub>o</sub> <sub>accura</sub>t<sub>e</sub>l<sub>y</sub> <sub>pre</sub>di<sub>c</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>ormance</sub> i<sub>n</sub> <sub>a</sub> <sub>new</sub>l<sub>y</sub> <sub>g</sub>i<sub>ven spec</sub>ifi<sub>c samp</sub>l<sub>e, we use</sub>d th<sub>e pre</sub>di<sub>c</sub>t<sub>e</sub>d $z _ { i }$ i<sub>n</sub> th<sub>e ex</sub>t<sub>erna</sub>l t<sub>es</sub>t <sub>se</sub>t t<sub>o separa</sub>t<sub>e a</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>group</sub> and a low-<sub>p</sub>erformance sub<sub>g</sub>rou<sub>p</sub> b<sub>y</sub> their median. We therefore com<sub>p</sub>ared the two ROC curves[11] of the two <sub>su</sub>b<sub>groups</sub> <sub>an</sub>d <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub>d th<sub>e</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d AUC <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f <sub>eac</sub>h <sub>su</sub>b<sub>group.</sub> If th<sub>e</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub>

F<sub>ur</sub>th<sub>ermore, we exam</sub>i<sub>ne</sub> th<sub>e assoc</sub>i<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween pre</sub>di<sub>c</sub>t<sub>e</sub>d l<sub>oca</sub>l <sub>per</sub>f<sub>ormance an</sub>d <sub>o</sub>b<sub>serve</sub>d <sub>mo</sub>d<sub>e</sub>l di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub> ti<sub>on</sub> b<sub>y</sub> <sub>compu</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>c</sub>l<sub>ass</sub>i<sub>ca</sub>l AUC <sub>o</sub>f th<sub>e</sub> PREVENT <sub>mo</sub>d<sub>e</sub>l f<sub>or</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>su</sub>b<sub>groups</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> <sub>percen</sub>til<sub>es</sub> <sub>o</sub>f th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d $z _ { i }$ (from the lowest 1% to 100%, with step size 1%).

## 4 Simulation studies

T<sub>o</sub> f<sub>ur</sub>th<sub>er</sub> <sub>va</sub>lid<sub>a</sub>t<sub>e</sub> <sub>our</sub> <sub>me</sub>th<sub>o</sub>d<sub>,</sub> <sub>we</sub> <sub>cons</sub>id<sub>ere</sub>d <sub>severa</sub>l di<sub>s</sub>ti<sub>nc</sub>t <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>scenar</sub>i<sub>os:</sub> t<sub>wo</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>examp</sub>l<sub>es</sub> <sub>an</sub>d <sub>one</sub> <sub>nega</sub>ti<sub>ve</sub> <sub>con</sub>t<sub>ro</sub>l <sub>examp</sub>l<sub>e.</sub>

## 4.1 Simulation Scenario 1: Identification of Subgroups with Heterogeneous Model performance (Positive Control)

Fi<sub>rs</sub>t<sub>, we assume</sub> th<sub>a</sub>t th<sub>ere</sub> i<sub>s a pre</sub>di<sub>c</sub>t<sub>or var</sub>i<sub>a</sub>bl<sub>e</sub> th<sub>a</sub>t <sub>separa</sub>t<sub>es</sub> th<sub>e w</sub>h<sub>o</sub>l<sub>e</sub> d<sub>a</sub>t<sub>a se</sub>t i<sub>n</sub>t<sub>o</sub> t<sub>wo</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>su</sub>b<sub>se</sub>t<sub>s</sub> $D _ { M }$ <sub>an</sub>d $D _ { N }$ <sub>.</sub> T<sub>wo</sub> bi<sub>nary</sub> <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on</sub> <sub>re</sub>f<sub>erence</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> $f _ { M }$ <sub>an</sub>d $f _ { N }$ <sub>are</sub> <sub>suc</sub>h th<sub>a</sub>t $f _ { i }$ h<sub>as</sub> <sub>more</sub> <sub>prec</sub>i<sub>se</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> th<sub>an</sub> th<sub>e o</sub>th<sub>er</sub> i<sub>n su</sub>b<sub>se</sub>t $S _ { i }$ where i can be either M or N.

W<sub>e assu</sub>m<sub>e</sub>d $x _ { 1 } , x _ { 2 } , x _ { 3 }$ i<sub>.</sub>i<sub>.</sub>d<sub>.,</sub> <sub>w</sub>h<sub>ere</sub> $x _ { 1 } , x _ { 2 } \sim N ( 0 , 1 )$ <sub>,</sub> <sub>an</sub>d $x _ { 3 } \sim U [ 0 , 1 ]$ <sub>as</sub> <sub>a</sub> l<sub>a</sub>t<sub>en</sub>t <sub>su</sub>b<sub>group</sub> id<sub>en</sub>tifi<sub>er.</sub>

T<sub>wo mo</sub>d<sub>e</sub>l<sub>s</sub> $f _ { M }$ <sub>an</sub>d $f _ { N }$ <sub>are</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as:</sub>

1<sub>.</sub> $f _ { M } ( x _ { 1 } , x _ { 2 } ) = x _ { 1 } - 1 0 x _ { 2 }$

2<sub>.</sub> $f _ { N } ( x _ { 1 } , x _ { 2 } ) = 1 0 x _ { 1 } ^ { 2 } x _ { 2 }$

S<sub>p</sub>lit th<sub>e</sub> <sub>w</sub>h<sub>o</sub>l<sub>e</sub> d<sub>a</sub>t<sub>a</sub> <sub>se</sub>t i<sub>n</sub>t<sub>o</sub> t<sub>wo</sub> <sub>su</sub>b<sub>groups:</sub> $D = D _ { M } \cup D _ { N }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub>

1<sub>.</sub> F<sub>or</sub> $D _ { M } x _ { 3 } < 0 . 5$ <sub>.</sub> F<sub>or</sub> <sub>eac</sub>h i<sub>ns</sub>t<sub>ance,</sub> <sub>genera</sub>t<sub>e</sub> t<sub>rue</sub> <sub>ou</sub>t<sub>come</sub> $y = 1$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> ${ p _ { M } } = \mathrm { l o g i t } ^ { - 1 } ( { f _ { M } ( x _ { 1 } , x _ { 2 } ) } )$ . Oth<sub>erw</sub>i<sub>se</sub> $y = 0$

2<sub>.</sub> F<sub>or</sub> $D _ { N } x _ { 3 } \ge 0 . 5$ . For each instance, generate true outcome y = 1 with probabilit<sub>y</sub> ${ p _ { N } } = \mathrm { l o g i t } ^ { - 1 } ( f _ { N } ( x _ { 1 } , x _ { 2 } ) )$ Oth<sub>erw</sub>i<sub>se</sub> $y = 0$

Construct a simulated training data set with 5000 instances and testing data set with 1000 instances, using the rule <sub>a</sub>b<sub>ove.</sub>

Tr<sub>a</sub>in t<sub>wo</sub> XGB<sub>oos</sub>t m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> $g _ { M } , g _ { N }$ <sub>on</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a</sub> <sub>se</sub>t<sub>,</sub> <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> i<sub>n</sub> th<sub>e</sub> <sub>w</sub>h<sub>o</sub>l<sub>e</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub>

$g _ { M }$ i<sub>s</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>on</sub> l<sub>og</sub>it<sub>-</sub>t<sub>rans</sub>f<sub>orme</sub>d l<sub>oca</sub>li<sub>ze</sub>d $\mathrm { L A U C } ( x _ { i } )$ <sub>o</sub>f <sub>re</sub>f<sub>erence mo</sub>d<sub>e</sub>l $f _ { M }$

$g _ { N }$ i<sub>s</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>on</sub> l<sub>og</sub>it<sub>-</sub>t<sub>rans</sub>f<sub>orme</sub>d l<sub>oca</sub>li<sub>ze</sub>d $\mathrm { L A U C } ( x _ { i } )$ <sub>o</sub>f <sub>re</sub>f<sub>erence mo</sub>d<sub>e</sub>l $f _ { N }$

Fi<sub>gure</sub> 1 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t $g _ { M }$ tends to predict higher logit $ { \left( \mathrm { L A U C } ( x _ { i } ) \right. }$ <sub>va</sub>l<sub>ues w</sub>h<sub>en</sub> $x _ { 3 } < 0 . 5$ <sub>w</sub>hil<sub>e</sub> $g _ { N }$ t<sub>en</sub>d<sub>s</sub> t<sub>o</sub> <sub>pre</sub>di<sub>c</sub>t hi<sub>g</sub>h<sub>er</sub> <sub>va</sub>l<sub>ues</sub> i<sub>n</sub> th<sub>e</sub> $x _ { 3 } \geq 0 . 5$ re<sub>g</sub><sup>i</sup>me.

R<sub>eca</sub>ll f<sub>rom</sub> th<sub>e s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on se</sub>t<sub>up</sub> th<sub>a</sub>t th<sub>e en</sub>ti<sub>re</sub> d<sub>a</sub>t<sub>a se</sub>t <sub>was sp</sub>lit i<sub>n</sub>t<sub>o</sub> t<sub>wo su</sub>b<sub>se</sub>t<sub>s</sub> $D _ { M } , D _ { N }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $f _ { M }$ <sub>p</sub>er<sup>f</sup>orms b<sub>e</sub>tt<sub>er</sub> th<sub>an</sub> $f _ { N }$ i<sub>n</sub> d<sub>a</sub>t<sub>a</sub> <sub>se</sub>t $D _ { M }$ <sub>,</sub> <sub>an</sub>d <sub>v</sub>i<sub>se</sub> <sub>versa.</sub> Th<sub>e</sub> <sub>sp</sub>lit b<sub>e</sub>t<sub>ween</sub> $D _ { M } , D _ { N }$ <sub>s</sub>i d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> $x _ { 3 } < 0 . 5$

![](images/5274ba157ec031383732de36cb2a4097db4f8220ef8833fb5a2fbe68a9f2f063.jpg)  
Figure 1. The comparison of predicted $\mathrm { l o g i t } ( \mathrm { L A U C } ( x _ { i } ) )$ <sub>va</sub>l<sub>ues</sub> b<sub>e</sub>t<sub>ween mo</sub>d<sub>e</sub>l $g _ { M }$ (Model M in legend) and model $g _ { N }$ (Model N in legend) in a simulated data set. Accordin<sub>g</sub> to the simulation desi<sub>g</sub>n in section $^ { 4 , }$ <sub>mo</sub>d<sub>e</sub>l $f _ { M }$ <sub>s</sub>h<sub>ou</sub>ld <sub>g</sub>i<sub>ve more prec</sub>i<sub>se pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>w</sub>h<sub>en</sub> $x _ { 3 } < 0 . 5$ th<sub>an mo</sub>d<sub>e</sub>l $f _ { N }$ <sub>, w</sub>hil<sub>e mo</sub>d<sub>e</sub>l $f _ { N }$ <sub>s</sub>h<sub>ou</sub>ld <sub>g</sub>i<sub>ve more prec</sub>i<sub>se pre</sub>di<sub>c</sub>ti<sub>on w</sub>h<sub>en</sub> $x _ { 3 } \geq 0 . 5$ $g _ { M }$ <sub>an</sub>d $g _ { N }$ <sub>a</sub>r<sub>e</sub> XGB<sub>oos</sub>t m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> t<sub>ra</sub>i<sub>ne</sub>d b<sub>ase</sub>d <sub>on</sub> th<sub>e we</sub>i<sub>g</sub>ht<sub>e</sub>d $\mathrm { l o g i t } ( \mathrm { L A U C } ( x _ { i } ) )$ <sub>o</sub>f $f _ { M }$ <sub>an</sub>d $f _ { N }$ <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> F<sub>rom</sub> th<sub>e</sub> fi<sub>gure</sub> it i<sub>s c</sub>l<sub>ear</sub> th<sub>a</sub>t <sub>mo</sub>d<sub>e</sub>l $g _ { M }$ h<sub>as</sub> l<sub>arger</sub> $\mathrm { l o g i t } ( \mathrm { L A U C } ( x _ { i } ) )$ ) than model $g _ { N }$ <sub>w</sub>h<sub>en</sub> $x _ { 3 } < 0 . 5$ <sub>w</sub>hil<sub>e mo</sub>d<sub>e</sub>l $g _ { N }$ has larger logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$ th<sub>an mo</sub>d<sub>e</sub>l $g _ { M }$ <sub>w</sub>h<sub>en</sub> $x _ { 3 } \geq 0 . 5$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>a</sub>li<sub>gns</sub> <sub>w</sub>ith <sub>our</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on.</sub>

This simulations illustrates that localized logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$ i<sub>s sens</sub>iti<sub>ve</sub> t<sub>o</sub> $x _ { 3 }$ th<sub>a</sub>t <sub>was use</sub>d t<sub>o separa</sub>t<sub>e</sub> $D _ { M }$ <sub>an</sub>d $D _ { N }$

## 4.2 Simulation Scenario 2: negative control experiment

To justify that our method works also well when there is no way to diferentiate high- and low-performance <sub>su</sub>b<sub>groups,</sub> <sub>we</sub> d<sub>es</sub>i<sub>gne</sub>d <sub>ano</sub>th<sub>er</sub> <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on.</sub>

Suppose that there is a dataset which has 6 variables $x _ { 1 } , x _ { 2 } , \ldots , x _ { 6 }$ <sub>an</sub>d <sub>one</sub> bi<sub>nary ou</sub>t<sub>come</sub> $y \in \{ 0 , 1 \}$ <sub>.</sub> Th<sub>e</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> f<sub>o</sub>ll<sub>ow</sub> th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on:</sub>

1. y follows Bernoulli distribution $y \sim \mathtt { B e r n } ( 0 . 5 )$

2<sub>.</sub> $x _ { 1 }$ f<sub>o</sub>ll<sub>ows con</sub>diti<sub>ona</sub>l <sub>norma</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $x _ { 1 } \sim { \mathcal { N } } ( A \cdot y , 1 )$ <sub>w</sub>ith <sub>a c</sub>h<sub>osen parame</sub>t<sub>er</sub> $A = 1 . 8 1 2 3 8 8$

3<sub>.</sub> $x _ { 2 } , x _ { 3 } , \ldots , x _ { 6 } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 )$

Th<sub>e re</sub>f<sub>erence mo</sub>d<sub>e</sub>l b<sub>e</sub>i<sub>ng eva</sub>l<sub>ua</sub>t<sub>e</sub>d i<sub>s</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a-genera</sub>ti<sub>ng mo</sub>d<sub>e</sub>l $\begin{array} { r } { f ( x _ { 1 } ) = \frac { 1 } { 1 + e ^ { - x _ { 1 } } } } \end{array}$ <sub>.</sub> S<sub>o</sub> th<sub>a</sub>t <sub>every</sub> b<sub>ase</sub>li<sub>ne var</sub>i<sub>a</sub>bl<sub>e</sub> <sup>exce</sup>p<sup>t</sup> $x _ { 1 }$ i<sub>s</sub> <sub>ran</sub>d<sub>om</sub> <sub>no</sub>i<sub>se.</sub>

We then applied our method and trained the XGBoost model g on $x 2 , \ldots , x _ { 6 } , y$ (excludin<sub>g</sub> $x _ { 1 } )$ <sub>,</sub> <sub>so</sub> th<sub>a</sub>t th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l

g is trained purely on the noise. The resulting ROC curves of high and low-performance subgroups identified by $g$ i<sub>n</sub> th<sub>e</sub> h<sub>o</sub>ld<sub>ou</sub>t t<sub>es</sub>t <sub>se</sub>t <sub>are s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 2<sub>.</sub>

![](images/eff3f306c312fb8fa8df05a64f12242b10009aa0881012efbae09b6d381332de.jpg)  
Figure 2. ROC curves com arison of low- and hi h- erformance sub rou s from the ne ative-control simulation 4.2. The XGBoost <sub>mo</sub>d<sub>e</sub>l <sub>use</sub>d t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>- an</sub>d l<sub>ow-per</sub>f<sub>ormance su</sub>b<sub>groups</sub> i<sub>s</sub> b<sub>ase</sub>d <sub>on pure no</sub>i<sub>se.</sub> Th<sub>e con</sub>fid<sub>ence</sub> b<sub>an</sub>d<sub>s o</sub>f th<sub>e</sub> t<sub>wo</sub> ROC <sub>curves</sub> over<sup>l</sup>a<sub>p</sub>, as ex<sub>p</sub>ecte<sup>d</sup>.

## 4.3 Simulation Scenario 3: Low Performance in the Overlap Region

W<sub>e se</sub>t <sub>up</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on:</sub>

1<sub>.</sub> $y$ f<sub>o</sub>ll<sub>ows</sub> B<sub>ernou</sub>lli di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $y \sim \mathtt { B e r n } ( 0 . 5 )$

2<sub>.</sub> $x _ { 1 }$ f<sub>o</sub>ll<sub>ows con</sub>diti<sub>ona</sub>l <sub>norma</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $x _ { 1 } \sim { \mathcal { N } } ( A \cdot y , 1 )$ <sub>w</sub>ith <sub>a</sub> <sub>c</sub>h<sub>osen</sub> <sub>parame</sub>t<sub>er</sub> $A = 1 . 1 9 0 2 3 2$

3<sub>.</sub> $x _ { 2 } \sim \mathcal { N } ( 0 . 1 7 7 7 1 , 1 )$

4<sub>.</sub> $x _ { 3 } \sim \mathcal { U } [ 0 , 1 ]$

This simulations setup is expected to produce unweighted AUC of 0.8. The actual unweighted AUC estimated in the test set is 0.8143

W<sub>e</sub> t<sub>ra</sub>i<sub>ne</sub>d th<sub>e</sub> XGB<sub>oos</sub>t <sub>mo</sub>d<sub>e</sub>l i<sub>n</sub> th<sub>e</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>se</sub>t <sub>an</sub>d d<sub>er</sub>i<sub>ve</sub>d th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub> <sub>an</sub>d l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>groups</sub> i<sub>n</sub> th<sub>e</sub> t<sub>es</sub>t <sub>se</sub>t<sub>.</sub> Th<sub>e</sub> ROC <sub>curves are a</sub>l<sub>so o</sub>bt<sub>a</sub>i<sub>ne</sub>d <sub>as</sub> Fi<sub>gure</sub> 3<sub>.</sub> Th<sub>e con</sub>fid<sub>ence</sub> l<sub>eve</sub>l i<sub>s se</sub>t t<sub>o</sub> b<sub>e</sub> 95%<sub>.</sub> Th<sub>e</sub> hi<sub>g</sub>h <sub>per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> h<sub>as</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y</sub> hi<sub>g</sub>h<sub>er</sub> ROC <sub>curve</sub> th<sub>an</sub> l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group,</sub> <sub>an</sub>d th<sub>e</sub> <sub>con</sub>fid<sub>ence</sub> bands are se<sub>p</sub>arable. The standard (unwei<sub>g</sub>hted) AUC value for low-<sub>p</sub>erformance sub<sub>g</sub>rou<sub>p</sub> is AUC=0.69, with $9 5 \% C \mathrm { T } { = } [ 0 . 6 6 ; 0 . 7 1 ]$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> l<sub>ower</sub> th<sub>an</sub> th<sub>e</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d AUC <sub>va</sub>l<sub>ue</sub> f<sub>or</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> $\mathrm { A U C = } 0 . 8 8$ <sub>,</sub> <sub>w</sub>ith $9 5 \% C [ = [ 0 . 8 6 - 0 . 9 0 ]$

![](images/67363cf4c564a6b0e5841e26886951e9a1ef4f8b543e7ebef17709da405632fa.jpg)  
Figure 3. ROC curves comparison of low- and high-performance subgroups of the simulation as in section 4.3. The test set is split into l<sub>ow- an</sub>d hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>groups accor</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e me</sub>di<sub>an o</sub>f th<sub>e pre</sub>di<sub>c</sub>ti<sub>ons o</sub>f th<sub>e</sub> t<sub>ra</sub>i<sub>ne</sub>d XGB<sub>oos</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> Th<sub>e con</sub>fid<sub>ence</sub> l<sub>eve</sub>l i<sub>s</sub> <sub>se</sub>t t<sub>o</sub> b<sub>e</sub> 95%<sub>.</sub> Th<sub>e</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> h<sub>as</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y</sub> hi<sub>g</sub>h<sub>er</sub> ROC <sub>curve</sub> th<sub>an</sub> l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group,</sub> <sub>an</sub>d th<sub>e</sub> confidence bands are se<sub>p</sub>arable. The standard (unwei<sub>g</sub>hted) AUC value for low-<sub>p</sub>erformance sub<sub>g</sub>rou<sub>p</sub> is AUC=0.69, with CI<sub>=</sub>0<sub>.</sub>66<sub>-</sub>0<sub>.</sub>71<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> l<sub>ower</sub> th<sub>an</sub> th<sub>e</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d AUC <sub>va</sub>l<sub>ue</sub> f<sub>or</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> $\mathrm { A U C = } 0 . 8 8$ <sub>,</sub> <sub>w</sub>ith $\scriptstyle \mathrm { C I = 0 . 8 6 - 0 . 9 0 }$ <sub>.</sub> Th<sub>e</sub> fi<sub>gure</sub> <sub>g</sub>i<sub>ves ano</sub>th<sub>er va</sub>lid<sub>a</sub>ti<sub>on</sub> f<sub>or our me</sub>th<sub>o</sub>d i<sub>n</sub> id<sub>en</sub>tif<sub>y</sub>i<sub>ng per</sub>f<sub>ormance su</sub>b<sub>groups o</sub>f th<sub>e cer</sub>t<sub>a</sub>i<sub>n g</sub>i<sub>ven mo</sub>d<sub>e</sub>l<sub>.</sub>

Alth<sub>oug</sub>h i<sub>n</sub> thi<sub>s</sub> <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>scenar</sub>i<sub>o</sub> d<sub>a</sub>t<sub>a</sub> <sub>was</sub> <sub>pro</sub>d<sub>uce</sub>d b<sub>y</sub> th<sub>e</sub> <sub>same</sub> d<sub>a</sub>t<sub>a-genera</sub>ti<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>s</sub>till id<sub>en</sub>tifi<sub>e</sub>d hi<sub>g</sub>h <sub>an</sub>d l<sub>ow per</sub>f<sub>ormance su</sub>b<sub>groups.</sub> T<sub>o exp</sub>l<sub>a</sub>i<sub>n</sub> thi<sub>s, we p</sub>l<sub>o</sub>tt<sub>e</sub>d i<sub>n gray</sub> hi<sub>s</sub>t<sub>ogram o</sub>f $x _ { 1 }$ i<sub>n</sub> low <sub>p</sub>erformance sub<sub>g</sub>rou<sub>p</sub> and in oran<sub>g</sub>e in hi<sub>g</sub>h <sub>p</sub>erformance sub<sub>g</sub>rou<sub>p</sub> (Fi<sub>g</sub>ure 4). It is notable that the l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> i<sub>s</sub> i<sub>n</sub> th<sub>e</sub> <sub>m</sub>iddl<sub>e</sub> <sub>o</sub>f t<sub>wo</sub> <sub>pea</sub>k<sub>s</sub> <sub>o</sub>f th<sub>e</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>groups.</sub> R<sub>emem</sub>b<sub>er</sub> th<sub>a</sub>t $x _ { 1 } \sim { \mathcal { N } } ( A \cdot y , 1 )$ <sub>w</sub>ith t<sub>wo norma</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on cen</sub>t<sub>ers aroun</sub>d $x _ { 1 } = 0$ <sub>an</sub>d $x _ { 1 } = A = 1 . 1 9 0 2 3 2$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore, aroun</sub>d $\textstyle x _ { 1 } = { \frac { A } { 2 } } = 0 . 5 9 5 1 1 6$ <sub>,</sub> t<sub>wo</sub> <sub>con</sub>diti<sub>ona</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> <sub>over</sub>l<sub>ap</sub> <sub>an</sub>d it i<sub>s</sub> h<sub>ar</sub>d t<sub>o</sub> <sub>separa</sub>t<sub>e</sub> <sub>even</sub>t<sub>s</sub> f<sub>rom</sub> <sub>non-even</sub>t<sub>s</sub> i<sub>n</sub> thi<sub>s</sub> <sub>reg</sub>i<sub>on.</sub> Hi<sub>g</sub>h <sub>per</sub>f<sub>ormance</sub> <sub>reg</sub>i<sub>on</sub> <sub>cons</sub>i<sub>s</sub>t<sub>s</sub> <sub>o</sub>f th<sub>e</sub> l<sub>e</sub>ft <sub>an</sub>d <sub>r</sub>i<sub>g</sub>ht b<sub>oun</sub>d<sub>ary</sub> <sub>o</sub>f $x _ { 1 }$ d<sub>oma</sub>i<sub>n</sub> <sub>w</sub>h<sub>ere</sub> <sub>even</sub>t<sub>s</sub> <sub>an</sub>d <sub>non-even</sub>t<sub>s</sub> <sub>are a</sub>l<sub>mos</sub>t <sub>per</sub>f<sub>ec</sub>tl<sub>y separa</sub>t<sub>e</sub>d<sub>.</sub> Thi<sub>s examp</sub>l<sub>e</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>a</sub>t <sub>a re</sub>f<sub>erence mo</sub>d<sub>e</sub>l <sub>w</sub>ith i<sub>mper</sub>f<sub>ec</sub>t di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns reg</sub>i<sub>ons w</sub>ith h<sub>e</sub>t<sub>erogenous per</sub>f<sub>ormance.</sub>

![](images/16cfde6cbeec7244d9570b400f12251698c25cef09436227630bfa4ce46dcfb2.jpg)  
Figure 4. The distribution of $x _ { 1 }$ <sub>o</sub>f l<sub>ow-</sub> <sub>an</sub>d hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>groups,</sub> id<sub>en</sub>tifi<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> XGB<sub>oos</sub>t <sub>mo</sub>d<sub>e</sub>l t<sub>ra</sub>i<sub>ne</sub>d i<sub>n</sub> th<sub>e</sub> <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on.</sub> Th<sub>e</sub> l<sub>ow-per</sub>f<sub>ormance su</sub>b<sub>group</sub> i<sub>s</sub> i<sub>n</sub> th<sub>e m</sub>iddl<sub>e o</sub>f t<sub>wo</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>groups, w</sub>hi<sub>c</sub>h <sub>a</sub>li<sub>gns w</sub>ith th<sub>e s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on se</sub>t <sub>up</sub> $x _ { 1 } \sim { \mathcal { N } } ( A \cdot y , 1 )$ <sub>w</sub>ith t<sub>wo norma</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons over</sub>l<sub>app</sub>i<sub>ng aroun</sub>d $\textstyle x _ { 1 } = { \frac { A } { 2 } } = 0 . 5 9 5 1 1 6$ <sub>.</sub> S<sub>o a</sub>t th<sub>e m</sub>iddl<sub>e po</sub>i<sub>n</sub>t it i<sub>s</sub> h<sub>ar</sub>d t<sub>o</sub> d<sub>ec</sub>id<sub>e</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a po</sub>i<sub>n</sub>t <sub>comes</sub> f<sub>rom</sub> th<sub>e even</sub>t d<sub>a</sub>t<sub>a</sub> $y = 1$ <sub>or non-even</sub>t d<sub>a</sub>t<sub>a</sub> $y = 0$

## 5 Disentangling Performance Drivers: Model Heterogeneity vs. Cohort Case-Mix

E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> <sub>r</sub>i<sub>s</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>v</sub>i<sub>a</sub> l<sub>oca</sub>li<sub>ze</sub>d di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on</sub> f<sub>unc</sub>ti<sub>ona</sub>l<sub>s-suc</sub>h <sub>as</sub> th<sub>e</sub> <sub>per-person</sub> LAUC<sub>,</sub> <sub>may</sub> <sub>revea</sub>l <sub>per</sub>f<sub>ormance</sub> <sub>gaps</sub> b<sub>e</sub>t<sub>ween</sub> hi<sub>g</sub>h<sub>-compe</sub>t<sub>ence</sub> $( S _ { 1 } )$ <sub>an</sub>d l<sub>ow-compe</sub>t<sub>ence</sub> $( S _ { 2 } )$ <sup>s</sup>u<sup>b</sup>g<sup>ro</sup>up<sup>s</sup> $( \Delta \mathrm { A U C } = \mathrm { A U C } _ { S _ { 1 } } -$ $\mathrm { A U C } _ { S _ { 2 } } > 0 )$ <sub>.</sub> H<sub>owever,</sub> <sub>re</sub>l<sub>y</sub>i<sub>ng</sub> <sub>so</sub>l<sub>e</sub>l<sub>y</sub> <sub>on</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>summary</sub> <sub>me</sub>t<sub>r</sub>i<sub>c</sub> lik<sub>e</sub> $\Delta \mathrm { L A U C } ( x _ { i } )$ <sub>may</sub> <sub>con</sub>fl<sub>a</sub>t<sub>e</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>ll<sub>y</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>p</sub>h<sub>enomena</sub> <sub>.</sub> A l<sub>oca</sub>li<sub>ze</sub>d d<sub>rop</sub> i<sub>n</sub> AUC d<sub>oes</sub> <sub>no</sub>t <sub>au</sub>t<sub>oma</sub>ti<sub>ca</sub>ll<sub>y</sub> i<sub>mp</sub>l<sub>y</sub> th<sub>a</sub>t th<sub>e</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>ll<sub>y</sub> fl<sub>awe</sub>d<sub>.</sub>

T<sub>o</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>ac</sub>ti<sub>ona</sub>bl<sub>e</sub> <sub>gu</sub>id<sub>e</sub>li<sub>nes</sub> f<sub>or</sub> <sub>mo</sub>d<sub>e</sub>l <sub>reme</sub>di<sub>a</sub>ti<sub>on</sub> <sub>versus</sub> <sub>co</sub>h<sub>or</sub>t <sub>se</sub>l<sub>ec</sub>ti<sub>on,</sub> <sub>we</sub> f<sub>orma</sub>li<sub>ze</sub> <sub>a</sub> t<sub>wo-ax</sub>i<sub>s</sub> di<sub>agnos</sub>ti<sub>c</sub> framework that separates Model Heterogeneity (whether the model performs worse than its assigned risks dictate) from Cohort Case-Mix / Risk Range Variance (whether the subgroup naturally lacks outcome-discriminating variance).

## 5.1 Local Underperformance Scenarios

E<sub>ven</sub> <sub>w</sub>h<sub>en</sub> <sub>a</sub> bi<sub>nary</sub> <sub>r</sub>i<sub>s</sub>k <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> <sub>correc</sub>tl<sub>y</sub> <sub>spec</sub>ifi<sub>e</sub>d <sub>across</sub> th<sub>e</sub> <sub>en</sub>ti<sub>re</sub> <sub>popu</sub>l<sub>a</sub>ti<sub>on,</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>co</sub>h<sub>or</sub>t <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub> <sub>may</sub> <sub>genera</sub>t<sub>e</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l l<sub>oca</sub>li<sub>ze</sub>d di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on</sub> <sub>gaps.</sub> W<sub>e</sub> <sub>c</sub>l<sub>ass</sub>if<sub>y</sub> l<sub>oca</sub>li<sub>ze</sub>d <sub>un</sub>d<sub>erper</sub>f<sub>ormance</sub> i<sub>n</sub>t<sub>o</sub> f<sub>our</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>scenar</sub>i<sub>os</sub> b<sub>ase</sub>d <sub>on</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on,</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>r</sub>i<sub>s</sub>k <sub>range,</sub> <sub>an</sub>d t<sub>rue</sub> <sub>mo</sub>d<sub>e</sub>l <sub>spec</sub>ifi<sub>ca</sub>ti<sub>on.</sub>

## 5.1.1 Scenario 1.1: Performance Degradation in Overlap Regions

Mechanism: In Simulation scenario 3 we observed that data-generating model may have low performance region <sub>w</sub>h<sub>ere</sub> <sub>r</sub>i<sub>s</sub>k <sub>scores</sub> <sub>over</sub>l<sub>ap</sub> <sub>an</sub>d hi<sub>g</sub>h <sub>per</sub>f<sub>ormance</sub> <sub>reg</sub>i<sub>on</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> <sub>o</sub>f th<sub>e</sub> <sub>oover</sub>l<sub>ap.</sub> Thi<sub>s</sub> <sub>occurs</sub> <sub>w</sub>h<sub>en</sub> b<sub>ase</sub>li<sub>ne</sub> <sub>mo</sub>d<sub>e</sub>l di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on</sub> f<sub>ar</sub> f<sub>rom</sub> i<sub>mper</sub>f<sub>ec</sub>t $( \mathrm { A U C } \ll 1 . 0 )$ <sub>,</sub> <sub>caus</sub>i<sub>ng</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>over</sub>l<sub>ap</sub> b<sub>e</sub>t<sub>ween</sub> <sub>even</sub>t <sub>an</sub>d <sub>non-even</sub>t <sub>r</sub>i<sub>s</sub>k di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> <sub>w</sub>ithi<sub>n</sub> <sub>spec</sub>ifi<sub>c</sub> <sub>p</sub>h<sub>eno</sub>t<sub>yp</sub>i<sub>c</sub> <sub>reg</sub>i<sub>ons.</sub>

Diagnosis & Action: Diagnosed by a wide risk range in low performance region paired low model heterogeneity $( \mathrm { A U C _ { \mathrm { o b s } } \ll A U C _ { \mathrm { e x p } } } )$ <sub>.</sub> Thi<sub>s</sub> <sub>re</sub>fl<sub>ec</sub>t<sub>s</sub> <sub>a</sub> <sub>genu</sub>i<sub>ne</sub> <sub>un</sub>d<sub>er-per</sub>f<sub>ormance</sub> <sub>o</sub>f th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> Withi<sub>n</sub> l<sub>ow-per</sub>f<sub>ormance</sub> <sub>reg</sub>i<sub>on</sub> t<sub>o</sub> di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>t<sub>e</sub> <sub>even</sub>t<sub>s</sub> f<sub>rom</sub> <sub>non-even</sub>t<sub>s</sub> b<sub>e</sub>tt<sub>er,</sub> <sub>a</sub>dditi<sub>ona</sub>l i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s</sub> <sub>requ</sub>i<sub>re</sub>d<sub>:</sub> <sub>o</sub>th<sub>er</sub> bi<sub>omar</sub>k<sub>er</sub> d<sub>a</sub>t<sub>a,</sub> i<sub>mag</sub>i<sub>ng</sub> etc.

## 5.1.2 Scenario 1.2: Homogeneous Covariate Clusters (Validation Cohort Artefact)

Mechanism: The low-performance subgroup $S _ { 2 }$ <sub>cons</sub>i<sub>s</sub>t<sub>s</sub> <sub>o</sub>f <sub>a</sub> d<sub>ense</sub> <sub>c</sub>l<sub>us</sub>t<sub>er</sub> <sub>o</sub>f <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s</sub> <sub>w</sub>ith <sub>near</sub>l<sub>y</sub> id<sub>en</sub>ti<sub>ca</sub>l covariate profiles X . Because the model assigns nearl<sub>y</sub> identical predicted risks $\hat { p } _ { i }$ t<sub>o</sub> <sub>s</sub>i<sub>m</sub>il<sub>ar</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l<sub>s,</sub> th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d <sub>r</sub>i<sub>s</sub>k <sub>range</sub> i<sub>s compresse</sub>d <sub>.</sub>

Diagnosis & Action: Characterized by a narrow covariate range, narrow predicted risk variance $( \sigma _ { \mathrm { l o g i t } ( \hat { p } ) | S _ { 2 } } ^ { 2 } \approx 0 )$ <sub>an</sub>d l<sub>ow</sub> <sub>mo</sub>d<sub>e</sub>l h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> $( \mathrm { A U C _ { \mathrm { o b s } } } \approx \mathrm { A U C _ { \mathrm { e x p } } } )$ <sub>.</sub> Thi<sub>s</sub> <sub>represen</sub>t<sub>s</sub> <sub>an</sub> <sub>ar</sub>tif<sub>ac</sub>t <sub>o</sub>f <sub>samp</sub>l<sub>e</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> cohort rather than a defect in the risk model: a model should assign identical risks to identical individuals . R<sub>e</sub>fitti<sub>ng</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>on</sub> th<sub>e</sub> <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> <sub>se</sub>t <sub>w</sub>ill <sub>no</sub>t i<sub>mprove</sub> di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on</sub> <sub>.</sub> Th<sub>e</sub> <sub>appropr</sub>i<sub>a</sub>t<sub>e</sub> <sub>reme</sub>d<sub>y</sub> i<sub>s</sub> t<sub>o</sub> <sub>enr</sub>i<sub>c</sub>h th<sub>e va</sub>lid<sub>a</sub>ti<sub>on co</sub>h<sub>or</sub>t’<sub>s covar</sub>i<sub>a</sub>t<sub>e</sub> di<sub>vers</sub>it<sub>y or measure nove</sub>l bi<sub>omar</sub>k<sub>ers</sub> t<sub>o re-</sub>i<sub>n</sub>t<sub>ro</sub>d<sub>uce r</sub>i<sub>s</sub>k <sub>var</sub>i<sub>ance .</sub>

## 5.1.3 Scenario 1.3: Covariate Diversity with Flat Risk Range (Biomarker Exhaustion)

Mechanism: Participants in $S _ { 2 }$ exhibit diverse covariate vectors X, <sub>y</sub>et the model maps these distinct profiles to <sub>a</sub> <sub>narrow,</sub> <sub>near</sub>l<sub>y</sub> <sub>un</sub>if<sub>orm</sub> b<sub>an</sub>d <sub>o</sub>f <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>r</sub>i<sub>s</sub>k <sub>.</sub> Thi<sub>s</sub> <sub>ar</sub>i<sub>ses</sub> <sub>un</sub>d<sub>er</sub> t<sub>wo</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>con</sub>diti<sub>ons:</sub>

label=(a) Omitted Biomarkers: The existing predictor set lacks the key variables required to stratify risk within thi<sub>s spec</sub>ifi<sub>c su</sub>b<sub>-popu</sub>l<sub>a</sub>ti<sub>on .</sub>

lbbel=(b) Latent Biomarker Clustering: An unobserved high-leverage biomarker (with true $\mathsf { A U C } \approx 1 . 0 )$ i<sub>s</sub> <sub>s</sub>t<sub>rong</sub>l<sub>y</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d $( R ^ { 2 } \approx 1 . 0 ) $ b<sub>y</sub> <sub>o</sub>b<sub>serve</sub>d <sub>covar</sub>i<sub>a</sub>t<sub>es,</sub> b<sub>u</sub>t th<sub>e</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>co</sub>h<sub>or</sub>t i<sub>na</sub>d<sub>ver</sub>t<sub>en</sub>tl<sub>y</sub> <sub>enro</sub>ll<sub>e</sub>d <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s s</sub>h<sub>ar</sub>i<sub>ng an</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>va</sub>l<sub>ue o</sub>f thi<sub>s</sub> l<sub>a</sub>t<sub>en</sub>t bi<sub>omar</sub>k<sub>er .</sub>

Diagnosis & Action: Characterized by reasonable covariate spread alongside a statistically flat risk variance $( R _ { U } < \delta$ via F-test) . If driven b<sub>y</sub> omitted predictors, the model must be augmented with novel clinical biomarkers

<sub>an</sub>d <sub>re</sub>fit <sub>.</sub> If d<sub>r</sub>i<sub>ven</sub> b<sub>y</sub> l<sub>a</sub>t<sub>en</sub>t bi<sub>omar</sub>k<sub>er c</sub>l<sub>us</sub>t<sub>er</sub>i<sub>ng,</sub> th<sub>e</sub> i<sub>ssue</sub> li<sub>es</sub> i<sub>n co</sub>h<sub>or</sub>t <sub>samp</sub>l<sub>e se</sub>l<sub>ec</sub>ti<sub>on,</sub> i<sub>mp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>a</sub>t id<sub>ea</sub>l <sub>va</sub>lid<sub>a</sub>ti<sub>on co</sub>h<sub>or</sub>t<sub>s s</sub>h<sub>ou</sub>ld <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n un</sub>if<sub>orm</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons over</sub> l<sub>a</sub>t<sub>en</sub>t <sub>r</sub>i<sub>s</sub>k f<sub>ac</sub>t<sub>ors.</sub>

## Test for Flat Predicted Risk

T<sub>o</sub> t<sub>es</sub>t <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>r</sub>i<sub>s</sub>k i<sub>s</sub> fl<sub>a</sub>t i<sub>n</sub> th<sub>e</sub> l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> $S$ <sub>compare</sub>d t<sub>o</sub> it<sub>s</sub> <sub>comp</sub>l<sub>emen</sub>t $S ^ { c }$ , we f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> t<sub>es</sub>t <sub>as:</sub>

$$
H _ { 0 } : \frac { \sigma _ { S } ^ { 2 } } { \sigma _ { S ^ { c } } ^ { 2 } } \geq \delta \quad \mathrm { v s . } \quad H _ { A } : \frac { \sigma _ { S } ^ { 2 } } { \sigma _ { S ^ { c } } ^ { 2 } } < \delta\tag{6}
$$

<sub>w</sub>h<sub>ere</sub> $S$ i<sub>s</sub> th<sub>e</sub> l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group,</sub> $S ^ { c }$ i<sub>s</sub> it<sub>s</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>,</sub> <sub>an</sub>d $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ <sub>represen</sub>t<sub>s</sub> th<sub>e non-</sub>fl<sub>a</sub>t<sub>ness</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>ra</sub>ti<sub>o.</sub>

At th<sub>e</sub> b<sub>oun</sub>d<sub>ary</sub> <sub>o</sub>f th<sub>e</sub> <sub>nu</sub>ll h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> $\begin{array} { r } { ( R = \frac { \sigma _ { S } ^ { 2 } } { \sigma _ { S ^ { c } } ^ { 2 } } = \delta ) } \end{array}$ , the test statistic scaled b<sub>y</sub> δ follows an F-distribution:

$$
\frac { s _ { S } ^ { 2 } } { \delta s _ { S ^ { c } } ^ { 2 } } \sim F _ { n _ { S } - 1 , n _ { S ^ { c } } - 1 }\tag{7}
$$

B<sub>ecause</sub> th<sub>e</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ve</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> <sub>pos</sub>it<sub>s</sub> <sub>a</sub> <sub>sma</sub>ll<sub>er</sub> <sub>var</sub>i<sub>ance</sub> i<sub>n</sub> $S ,$ this is a lower-tail test. The p-value is given b<sub>y</sub>:

$$
p = P \left( F _ { n _ { S } - 1 , n _ { S } c - 1 } \leq { \frac { s _ { S } ^ { 2 } } { \delta s _ { S } ^ { 2 } c } } \right)\tag{8}
$$

We reject $H _ { 0 }$ at significance level α when:

$$
{ \frac { s _ { S } ^ { 2 } } { \delta s _ { S ^ { c } } ^ { 2 } } } \leq F _ { \alpha ; n _ { S } - 1 , n _ { S ^ { c } } - 1 }\tag{9}
$$

E<sub>qu</sub>i<sub>va</sub>l<sub>en</sub>tl<sub>y,</sub> <sub>we</sub> <sub>can</sub> <sub>cons</sub>t<sub>ruc</sub>t <sub>a</sub> $1 0 0 ( 1 - \alpha ) \%$ <sub>one-s</sub>id<sub>e</sub>d <sub>upper con</sub>fid<sub>ence</sub> b<sub>oun</sub>d $R _ { U }$ f<sub>or</sub> th<sub>e</sub> t<sub>rue var</sub>i<sub>ance ra</sub>ti<sub>o</sub> $\begin{array} { r } { R = \frac { \sigma _ { S } ^ { 2 } } { \sigma _ { S ^ { c } } ^ { 2 } } ; } \end{array}$ :

$$
R _ { U } = \frac { \hat { R } } { F _ { \alpha ; n _ { S } - 1 , n _ { S } c - 1 } }\tag{10}
$$

and reject $H _ { 0 }$ if $R _ { U } < \delta$

D<sub>er</sub>i<sub>va</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> t<sub>es</sub>t i<sub>s</sub> <sub>prov</sub>id<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> A<sub>ppen</sub>di<sub>x.</sub>

## 5.1.4 Scenario 2: Genuine Local Model Misspecification (Structural Failure)

Mechanism: The baseline model is structurally misspecified within subgroup $S _ { 2 }$ d<sub>ue</sub> t<sub>o</sub> <sub>om</sub>itt<sub>e</sub>d <sub>non-</sub>li<sub>near</sub>iti<sub>es,</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ng</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> t<sub>erms,</sub> <sub>or</sub> i<sub>ncorrec</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> <sub>we</sub>i<sub>g</sub>ht<sub>s</sub> <sub>spec</sub>ifi<sub>c</sub> t<sub>o</sub> th<sub>a</sub>t <sub>p</sub>h<sub>eno</sub>t<sub>yp</sub>i<sub>c</sub> <sub>reg</sub>i<sub>on</sub> <sub>.</sub> E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons</sub>

<sub>un</sub>d<sub>er</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> <sub>severe</sub> di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on</sub> <sub>co</sub>ll<sub>apse</sub> $( \mathrm { e } . \mathrm { g } . , \Delta \mathrm { A U C } \geq + 0 . 3 5 )$

Diagnosis & Action: Diagnosed when $S _ { 2 }$ <sub>prov</sub>id<sub>es amp</sub>l<sub>e r</sub>i<sub>s</sub>k <sub>var</sub>i<sub>ance</sub> $( \sigma _ { \mathrm { l o g i t } ( \hat { p } ) | S _ { 2 } } ^ { 2 } \gg 0 )$ <sub>ye</sub>t <sub>ex</sub>hibit<sub>s a s</sub>i<sub>gn</sub>ifi<sub>can</sub>t excess r<sup>i</sup>s<sup>k</sup> <sub>g</sub>a<sub>p</sub> $( \mathrm { A U C _ { \mathrm { e x p } } - A U C _ { \mathrm { o b s } } \gg 0 ) }$ <sub>.</sub> Thi<sub>s</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>es unam</sub>bi<sub>guous mo</sub>d<sub>e</sub>l f<sub>a</sub>il<sub>ure .</sub> Th<sub>e reme</sub>d<sub>y requ</sub>i<sub>res</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>mo</sub>d<sub>e</sub>l <sub>re</sub>fitti<sub>ng,</sub> l<sub>oca</sub>li<sub>ze</sub>d <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on,</sub> <sub>or</sub> th<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub> <sub>o</sub>f <sub>su</sub>b<sub>group-spec</sub>ifi<sub>c</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> t<sub>erms</sub> <sub>.</sub>

## 6 Application: Localized Discrimination of the PREVENT Model in National Health and Nutrition Examination Survey (NHANES) and External Validation in the UK Biobank

A<sub>s</sub> <sub>a</sub> <sub>prac</sub>ti<sub>ca</sub>l <sub>examp</sub>l<sub>e</sub> <sub>we</sub> <sub>app</sub>l<sub>y</sub> <sub>propose</sub>d LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow</sub> <sub>per</sub>f<sub>ormance</sub> <sub>reg</sub>i<sub>ons</sub> <sub>o</sub>f PREVENT m<sub>o</sub>d<sub>e</sub>l<sub>.</sub> Am<sub>e</sub>ri<sub>ca</sub>n H<sub>ea</sub>rt A<sub>ssoc</sub>i<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd Am<sub>e</sub>ri<sub>ca</sub>n C<sub>o</sub>ll<sub>ege</sub> <sub>o</sub>f C<sub>a</sub>rd<sub>o</sub>l<sub>ogy</sub> <sub>cu</sub>rr<sub>e</sub>ntl<sub>y</sub> <sub>use</sub> PREVENT <sub>mo</sub>d<sub>e</sub>l t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> <sub>pa</sub>ti<sub>en</sub>t<sub>s</sub> <sub>w</sub>h<sub>o</sub> <sub>wou</sub>ld b<sub>ene</sub>fit f<sub>rom</sub> li<sub>p</sub>id <sub>an</sub>d bl<sub>oo</sub>d <sub>pressure-</sub>l<sub>ower</sub>i<sub>ng</sub> t<sub>rea</sub>t<sub>men</sub>t<sub>.</sub> PREVENT <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>use</sub>d f<sub>or</sub> <sub>m</sub>illi<sub>ons</sub> <sub>o</sub>f <sub>pa</sub>ti<sub>en</sub>t<sub>s.</sub> W<sub>e</sub> t<sub>ra</sub>i<sub>n</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> i<sub>n</sub> NHANES d<sub>a</sub>t<sub>a,</sub> <sub>va</sub>lid<sub>a</sub>t<sub>e</sub> i<sub>n</sub> h<sub>o</sub>ld <sub>ou</sub>t t<sub>es</sub>t <sub>su</sub>b<sub>se</sub>t <sub>an</sub>d <sub>ex</sub>t<sub>erna</sub>ll<sub>y</sub> <sub>va</sub>lid<sub>a</sub>t<sub>e</sub> i<sub>n</sub> UK Bi<sub>o</sub>b<sub>an</sub>k<sub>.</sub>

## 6.1 NHANES 1999–2010 Cohort Setup and Preprocessing

National Health and Nutrition Surve (NHANES) collects data on about 5, 000 adults and children in communities across the United States [12]. We used the c<sub>y</sub>cle ran<sub>g</sub>e 1999-2010 from NHANES data, with linked National Death Index data throu<sub>g</sub>h 2019 (N=17,748). We used CVD death as an outcome (N=756). Data were randoml<sub>y</sub> split into training (70%) and test (30%) subsets stratified b<sub>y</sub> categories of PREVENT predicted risk using the Python package scikit-learn[13] . We excluded participants outside of PREVENT predictors’ range, with prior CVD<sub>,</sub> <sub>w</sub>h<sub>o</sub> <sub>a</sub>r<sub>e</sub> <sub>p</sub>r<sub>eg</sub>n<sub>a</sub>nt<sub>,</sub> <sub>a</sub>nd <sub>w</sub>ith in<sub>co</sub>m<sub>p</sub>l<sub>e</sub>t<sub>e</sub> <sub>se</sub>t <sub>o</sub>f PREVENT <sub>p</sub>r<sub>e</sub>di<sub>c</sub>t<sub>o</sub>r <sub>va</sub>ri<sub>a</sub>bl<sub>es.</sub> W<sub>e</sub> <sub>app</sub>li<sub>e</sub>d in<sub>pu</sub>t r<sub>a</sub>n<sub>ges</sub> according to the R package preventr[14], as shown in Figure 9. Summary statistics of demographics and baseline <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub> i<sub>n</sub> NHANES <sub>cyc</sub>l<sub>es</sub> 1999<sub>-</sub>2010 <sub>are s</sub>h<sub>own</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 2 <sub>an</sub>d 3<sub>.</sub> T<sub>o a</sub>dd<sub>ress m</sub>i<sub>ss</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a</sub> i<sub>n var</sub>i<sub>a</sub>bl<sub>es</sub> <sub>no</sub>t <sub>use</sub>d b<sub>y</sub> PREVENT <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>we</sub> filt<sub>ere</sub>d <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s</sub> <sub>w</sub>ith <sub>more</sub> th<sub>an</sub> 40% <sub>o</sub>f b<sub>ase</sub>li<sub>ne</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ng;</sub> <sub>we</sub> <sub>a</sub>l<sub>so</sub> <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d b<sub>ase</sub>li<sub>ne var</sub>i<sub>a</sub>bl<sub>es w</sub>ith <sub>more</sub> th<sub>an</sub> 90% <sub>m</sub>i<sub>ss</sub>i<sub>ng.</sub> Aft<sub>er</sub> th<sub>a</sub>t<sub>, we u</sub>tili<sub>ze</sub>d M<sub>u</sub>lti<sub>p</sub>l<sub>e</sub> I<sub>mpu</sub>t<sub>a</sub>ti<sub>on</sub> b<sub>y</sub> Ch<sub>a</sub>i<sub>ne</sub>d Equations (MICE) with the P<sub>y</sub>thon <sub>p</sub>acka<sub>g</sub>e miceforest to trainin<sub>g</sub> and testin<sub>g</sub> subsets. The Gower distances, <sub>we</sub>i<sub>g</sub>ht<sub>s,</sub> $\mathrm { L A U C } ( x _ { i } )$ <sub>an</sub>d $z _ { i } = \mathrm { l o g i t } ( \mathrm { L A U C } ( x _ { i } ) )$ <sub>were</sub> <sub>compu</sub>t<sub>e</sub>d i<sub>n</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a</sub> <sub>se</sub>t<sub>.</sub> W<sub>e</sub> th<sub>en</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>an</sub> XGB<sub>oos</sub>t model with in<sub>p</sub>ut variables includin<sub>g</sub> PREVENT <sub>p</sub>redictors and other baseline variables (shown in Table 2 and 3). H<sub>yperparame</sub>t<sub>ers</sub> <sub>were</sub> t<sub>une</sub>d <sub>v</sub>i<sub>a</sub> <sub>ran</sub>d<sub>om</sub> <sub>searc</sub>h <sub>w</sub>ith 5<sub>-</sub>f<sub>o</sub>ld <sub>cross-va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>over</sub> 100 it<sub>era</sub>ti<sub>ons,</sub> <sub>exp</sub>l<sub>or</sub>i<sub>ng</sub> <sub>ranges</sub> for learnin<sub>g</sub> rate (0.1–0.3), maximum tree de<sub>p</sub>th (3–10), subsam<sub>p</sub>le ratio (0.5–1.0), column subsam<sub>p</sub>le ratio (0.5–1.0), and re<sub>g</sub>ularization <sub>p</sub>arameters $( \gamma , \alpha , \lambda )$ <sub>.</sub> E<sub>ar</sub>l<sub>y</sub> <sub>s</sub>t<sub>opp</sub>i<sub>ng</sub> <sub>w</sub>ith 20 <sub>roun</sub>d<sub>s</sub> <sub>o</sub>f <sub>no</sub> i<sub>mprovemen</sub>t <sub>was</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> <sub>preven</sub>t <sub>over</sub>fitti<sub>ng.</sub> Th<sub>e</sub> fi<sub>na</sub>l <sub>mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>ormance</sub> <sub>was</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>on</sub> th<sub>e</sub> t<sub>es</sub>t <sub>se</sub>t <sub>us</sub>i<sub>ng</sub> $R ^ { 2 }$ <sub>an</sub>d <sub>roo</sub>t <sub>mean</sub> <sub>square</sub>d error (RMSE). After 5-Fold Cross-Validation h<sub>yp</sub>er<sub>p</sub>arameter tunin<sub>g</sub>, we selected the best <sub>p</sub>arameters.

The model used the tree booster with the objective function znoteThe mean squared error (MSE) loss is convex, <sub>an</sub>d XGB<sub>oos</sub>t <sub>a</sub>dd<sub>s</sub> <sub>an</sub> $L _ { 2 }$ regularization term, making the overall objective strongly convex and thus ensuring the conver<sub>g</sub>ence of the trainin<sub>g</sub> loss [15]. However, due to the <sub>g</sub>reed<sub>y</sub> al<sub>g</sub>orithms em<sub>p</sub>lo<sub>y</sub>ed for both node s<sub>p</sub>littin<sub>g</sub> and se<sub>q</sub>uential tree addition [16][17], the o<sub>p</sub>timization t<sub>yp</sub>icall<sub>y</sub> conver<sub>g</sub>es to a local minimum within the constrained t<sub>ree-</sub>f<sub>unc</sub>ti<sub>on space ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>m</sub>i<sub>n</sub>i<sub>mum..</sub> T<sub>o come up w</sub>ith th<sub>e mos</sub>t <sub>pars</sub>i<sub>mon</sub>i<sub>ous</sub> XGB<sub>oos</sub>t <sub>mo</sub>d<sub>e</sub>l<sub>, we</sub> <sub>remove</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>ors</sub> th<sub>a</sub>t <sub>pro</sub>d<sub>uce</sub> <sub>sma</sub>ll<sub>er</sub> th<sub>an</sub> 10% <sub>ga</sub>i<sub>n</sub> <sub>o</sub>f th<sub>e</sub> <sub>s</sub>t<sub>ronges</sub>t <sub>pre</sub>di<sub>c</sub>t<sub>or.</sub> Th<sub>e</sub> R<sub>oo</sub>t<sub>-</sub>M<sub>ean-</sub>S<sub>quare-</sub>E<sub>rror</sub> (RMSE) of final XGBoost model was 0.0170 and the R2 score of 0.9713 in the test set.

<table><tr><td>Subgroup</td><td>Total Number</td><td>CVD death number</td><td>mean classical AUC value</td></tr><tr><td>Low perforamance</td><td>2663</td><td>190</td><td>0.780</td></tr><tr><td>High performance</td><td>2663</td><td>37</td><td>0.880</td></tr></table>

Table 1. Total sizes, CVD death event numbers and mean AUC values of high and low performance subgroups from test data set

T<sub>ra</sub>i<sub>ne</sub>d XGB<sub>oos</sub>t <sub>mo</sub>d<sub>e</sub>l <sub>was</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> t<sub>es</sub>t <sub>su</sub>b<sub>se</sub>t <sub>w</sub>hi<sub>c</sub>h <sub>was</sub> <sub>sp</sub>lit i<sub>n</sub>t<sub>o</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow</sub> <sub>per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> <sub>a</sub>t th<sub>e</sub> <sub>me</sub>di<sub>an</sub> <sub>o</sub>f <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d $z _ { i } = \mathrm { l o g i t } ( \mathrm { L A U C } ( x _ { i } ) )$ <sub>.</sub> C<sub>o</sub>n<sub>ve</sub>nti<sub>o</sub>n<sub>a</sub>l ROC <sub>cu</sub>r<sub>ves</sub> <sub>we</sub>r<sub>e</sub> <sub>p</sub>l<sub>o</sub>tt<sub>e</sub>d <sub>sepa</sub>r<sub>a</sub>t<sub>e</sub>l<sub>y</sub> in <sub>eac</sub>h <sub>g</sub>r<sub>oup</sub> <sub>an</sub>d <sub>conven</sub>ti<sub>ona</sub>l AUC <sub>was es</sub>ti<sub>ma</sub>t<sub>e</sub>d f<sub>or eac</sub>h<sub>.</sub>

## 6.2 Identification of PREVENT Competence Regions in NHANES

A high and a low performance subgroup were identified b<sub>y</sub> splitting the test set in halves, with 50% percentile of XGBoost prediction logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$ <sub>.</sub> T<sub>wo</sub> ROC <sub>cu</sub>r<sub>ves</sub> in th<sub>e</sub> t<sub>wo su</sub>b<sub>g</sub>r<sub>oups we</sub>r<sub>e</sub> dr<sub>aw</sub>n <sub>as</sub> in Fi<sub>gu</sub>r<sub>e</sub> 5<sub>.</sub>

The conventional AUC values in high and low performance subgroups are 0.88 95%CI[.82; .93] and 0.78 95%CI[.69; 77] res<sub>p</sub>ectivel<sub>y</sub>, shown in Table 1 and 5 and 7. The two ROC curves shown in Fi<sub>g</sub>ure 5 verif di<sub>rec</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> <sub>separa</sub>ti<sub>on</sub> <sub>o</sub>f hi<sub>g</sub>h<sub>-</sub> <sub>an</sub>d l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>groups</sub> id<sub>en</sub>tifi<sub>e</sub>d b<sub>y</sub> <sub>our</sub> <sub>propose</sub>d LOCATOR al<sub>g</sub>orithm in holdout test subset (internal validation). The corres<sub>p</sub>ondin<sub>g</sub> statistics of size, CVD death number and <sub>mean</sub> <sub>conven</sub>ti<sub>ona</sub>l AUC <sub>va</sub>l<sub>ue</sub> i<sub>n</sub> <sub>eac</sub>h <sub>su</sub>b<sub>group</sub> <sub>are</sub> <sub>s</sub>h<sub>own</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 1<sub>.</sub> W<sub>e</sub> <sub>p</sub>l<sub>o</sub>tt<sub>e</sub>d <sub>conven</sub>ti<sub>ona</sub>l AUC i<sub>n</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow</sub> <sub>per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>groups</sub> <sub>versus</sub> <sub>percen</sub>t <sub>ass</sub>i<sub>gne</sub>d t<sub>o</sub> l<sub>ow</sub> <sub>per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> i<sub>n</sub> Fi<sub>gure</sub> 6<sub>.</sub>

A <sub>ser</sub>i<sub>es o</sub>f <sub>conven</sub>ti<sub>ona</sub>l AUC <sub>va</sub>l<sub>ues</sub> i<sub>s p</sub>l<sub>o</sub>tt<sub>e</sub>d i<sub>n</sub> Fi<sub>gure</sub> 6 <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> th<sub>e percen</sub>t i<sub>n</sub> th<sub>e</sub> l<sub>owes</sub>t <sub>per</sub>f<sub>ormance</sub> <sup>s</sup>u<sup>b</sup>g<sup>ro</sup>up.

## 6.3 External Validation of PREVENT Competence Regions in UK Biobank

T<sub>o</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> <sub>ou</sub>t<sub>-o</sub>f<sub>-samp</sub>l<sub>e</sub> <sub>genera</sub>li<sub>za</sub>bilit<sub>y</sub> <sub>o</sub>f id<sub>en</sub>tifi<sub>e</sub>d <sub>compe</sub>t<sub>ence</sub> <sub>reg</sub>i<sub>ons,</sub> <sub>we</sub> <sub>app</sub>li<sub>e</sub>d t<sub>ra</sub>i<sub>ne</sub>d XGB<sub>oos</sub>t model in UK Biobank - a lar<sub>g</sub>e <sub>p</sub>ros<sub>p</sub>ective cohort (N=355,122). Data <sub>p</sub>re-<sub>p</sub>rocessin<sub>g</sub> and harmonization of UK Bi<sub>o</sub>b<sub>an</sub>k <sub>w</sub>ith NHANES d<sub>a</sub>t<sub>a,</sub> d<sub>emograp</sub>hi<sub>cs</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub> <sub>an</sub>d C<sub>onsor</sub>t di<sub>agram</sub> <sub>are</sub> <sub>prov</sub>id<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> A<sub>ppen</sub>di<sub>x.</sub>

![](images/479b7e099ece69f678a30595ffc9cc5643eb677e7ec503eec72af0f7bf816589.jpg)

![](images/faac2002f3571237a3c2936e1d7c1a975aea7f0dd7fe2bc8877523a1eaf63491.jpg)

![](images/a213e78f2d28050a4fa0af56f628db944a0aec4e28997e04dd7aa2ebfba3e6b6.jpg)  
Figure 5. The three lots show the ca abilit of diferent XGBoost models with maximum delta $( \mathrm { A U C } _ { i }$ diference), 70th ercentile d<sub>e</sub>lt<sub>a, an</sub>d <sub>me</sub>di<sub>an</sub> d<sub>e</sub>lt<sub>a.</sub> Th<sub>e</sub> t<sub>wo</sub> ROC <sub>curves</sub> i<sub>n a</sub> hi h<sub>- an</sub>d <sub>a</sub> l<sub>ow- er</sub>f<sub>ormance su</sub>b <sub>rou were</sub> id<sub>en</sub>tifi<sub>e</sub>d b <sub>s</sub> litti<sub>n a</sub>b<sub>ove an</sub>d b<sub>e</sub>l<sub>ow</sub> the median of XGBoost predictions of logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$ . <sup>S</sup>u<sup>b</sup><sub>g</sub>rou<sub>p</sub> $0 \% - 5 0 \%$ <sub>correspon</sub>d<sub>s</sub> t<sub>o</sub> l<sub>ow-per</sub>f<sub>ormance su</sub>b<sub>group an</sub>d Subgroup 50% − 100% corresponds to high-performance subgroup. The two ROC curves show separation, with AUC values shown in the legend significantl<sub>y</sub> diferent as expected. With confidence level of 95%, the confidence bands are shown in the figure and the <sub>con</sub>fid<sub>ence</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s o</sub>f AUC <sub>va</sub>l<sub>ues are s</sub>h<sub>own</sub> i<sub>n</sub> th<sub>e</sub> l<sub>egen</sub>d<sub>.</sub>

![](images/0346886a0eeebbe4298d96d7263f51c73607126d5bf2cf1370e2bba04881b29f.jpg)  
Figure 6. AUC with respect to the fraction assigned to the low-and high-performance subgroup in the NHANES data.

![](images/2d04372cae8f99c14418229b4f5eaa6a8af4c33b9515e3b0fabaaf9a6d598e43.jpg)

C<sub>ompe</sub>t<sub>ence</sub> <sub>reg</sub>i<sub>ons</sub> l<sub>earne</sub>d i<sub>n</sub> th<sub>e</sub> NHANES d<sub>a</sub>t<sub>a</sub> <sub>were</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> UK Bi<sub>o</sub>b<sub>an</sub>k <sub>an</sub>d <sub>conven</sub>ti<sub>ona</sub>l AUC <sub>was</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub>d i<sub>n</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow</sub> <sub>per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>groups</sub> <sub>us</sub>i<sub>ng</sub> CVD d<sub>ea</sub>th <sub>as</sub> <sub>an</sub> <sub>ou</sub>t<sub>come.</sub> AUC i<sub>n</sub> th<sub>e</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>g</sub>r<sub>oup</sub> <sub>was</sub> 0<sub>.</sub>761 <sub>w</sub>ith 95CI 0<sub>.</sub>742<sub>-</sub>0<sub>.</sub>781 <sub>a</sub>nd 0<sub>.</sub>708 <sub>w</sub>ith 95CI 0<sub>.</sub>697<sub>-</sub>0<sub>.</sub>719 in th<sub>e</sub> l<sub>ow-pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> <sub>su</sub>b<sub>g</sub>r<sub>oup</sub> i<sub>n</sub> th<sub>e</sub> UK Bi<sub>o</sub>b<sub>an</sub>k<sub>.</sub> Th<sub>ese</sub> <sub>num</sub>b<sub>ers</sub> <sub>s</sub>h<sub>ow</sub> <sub>m</sub>ild <sub>a</sub>tt<sub>enua</sub>ti<sub>on,</sub> b<sub>u</sub>t th<sub>e</sub> 95CI<sub>s</sub> d<sub>o</sub> <sub>no</sub>t <sub>over</sub>l<sub>ap</sub> <sub>an</sub>d <sub>are</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> <sub>.</sub>05 <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>s</sub>i<sub>gn</sub>ifi<sub>cance</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub>

![](images/8c6f3e9203ccef9597372802749ca3f4dc0180be7f2f1b546fe5b2f72e5aff12.jpg)

![](images/2f2b644d8346a358cff5c604f19f68004b7770feba7176763a79e4c181f27d47.jpg)

![](images/c20d2dbf27f67663b18f98661de334ab4922e095c0c182a61aa7ef9c9e61109e.jpg)  
Figure 7. External Validaton in UK Biobank. The two ROC curves of a high- and a low-performance subgroup were identified by splitting above and below the median of XGBoost predictions of logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$ <sup>S</sup>u<sup>b</sup><sub>g</sub>rou<sub>p</sub> $0 \% - 5 0 \%$ corres<sub>p</sub>on<sup>d</sup>s to l<sub>ow-pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce su</sub>b<sub>g</sub>r<sub>oup a</sub>nd S<sub>u</sub>b<sub>g</sub>r<sub>oup</sub> $5 0 \% - 1 0 0 \%$ <sub>co</sub>rr<sub>espo</sub>nd<sub>s</sub> t<sub>o</sub> hi<sub>g</sub>h<sub>-pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce su</sub>b<sub>g</sub>r<sub>oup.</sub> Th<sub>e</sub> t<sub>wo</sub> ROC <sub>cu</sub>r<sub>ves s</sub>h<sub>ow</sub> se aration with AUC values shown in the le end si nificantl diferent as ex ected. With confidence level of 95% the confidence b<sub>an</sub>d<sub>s are s</sub>h<sub>own</sub> i<sub>n</sub> th<sub>e</sub> fi<sub>gure an</sub>d th<sub>e con</sub>fid<sub>ence</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s o</sub>f AUC <sub>va</sub>l<sub>ues are s</sub>h<sub>own</sub> i<sub>n</sub> th<sub>e</sub> l<sub>egen</sub>d<sub>.</sub>

## 6.4 Identification of Variables associaited with high and low performance regions with SHAP analysis

Th<sub>e</sub> SHAP <sub>summary</sub> <sub>grap</sub>h i<sub>n</sub> Fi<sub>gure</sub> 8 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e</sub> <sub>mos</sub>t <sub>cr</sub>iti<sub>ca</sub>l <sub>per</sub>f<sub>ormance</sub> d<sub>r</sub>i<sub>vers</sub> <sub>are</sub> <sub>age,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> <sub>sex,</sub> t<sub>a</sub>ki<sub>ng an</sub>tih<sub>yper</sub>t<sub>ens</sub>i<sub>ve me</sub>di<sub>ca</sub>ti<sub>ons a</sub>t b<sub>ase</sub>li<sub>ne, curren</sub>t <sub>smo</sub>ki<sub>ng, an</sub>d th<sub>e use o</sub>f <sub>s</sub>t<sub>a</sub>ti<sub>n me</sub>di<sub>ca</sub>ti<sub>on.</sub> SHAP <sub>summary</sub> <sub>p</sub>l<sub>o</sub>t <sub>was</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d f<sub>rom</sub> th<sub>e</sub> t<sub>es</sub>t d<sub>a</sub>t<sub>ase</sub>t<sub>,</sub> <sub>as</sub> i<sub>n</sub> Fi<sub>g</sub> 8<sub>.</sub> It <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>o</sub>ld<sub>er</sub> <sub>age,</sub> bl<sub>oo</sub>d <sub>pressure</sub> t<sub>rea</sub>t<sub>men</sub>t<sub>,</sub> b<sub>e</sub>i<sub>ng</sub> <sub>ma</sub>l<sub>e,</sub> <sub>smo</sub>ki<sub>ng,</sub> <sub>a</sub> l<sub>ow</sub> <sub>pover</sub>t<sub>y</sub> i<sub>ncome</sub> <sub>ra</sub>ti<sub>o,</sub> t<sub>a</sub>ki<sub>ng</sub> <sub>s</sub>t<sub>a</sub>ti<sub>n</sub> <sub>me</sub>di<sub>ca</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>a</sub> hi<sub>s</sub>t<sub>ory</sub> <sub>o</sub>f t<sub>ype</sub> 2 di<sub>a</sub>b<sub>e</sub>t<sub>es</sub> <sub>an</sub>d <sub>cancer</sub> <sub>are</sub> <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith l<sub>ow</sub> <sub>per</sub>f<sub>ormance</sub> <sub>o</sub>f th<sub>e</sub> PREVENT <sub>mo</sub>d<sub>e</sub>l<sub>.</sub>

![](images/0e40fc3af19facc568b98736eb36cc952ce7f555f7b3e348363293b5b8ca0355.jpg)  
Figure 8. SHAP summary plot. the most critical variable is age, followed by sex, bp tx(now taking prescribed medicine), smoking and <sub>s</sub>t<sub>a</sub>ti<sub>n me</sub>di<sub>ca</sub>ti<sub>on use.</sub>

## 7 Discussion

I<sub>n</sub> thi<sub>s wor</sub>k<sub>, we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d th<sub>e</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m an</sub>d th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d LAUC f<sub>unc</sub>ti<sub>ona</sub>l t<sub>o a</sub>dd<sub>ress</sub> th<sub>e</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l li<sub>m</sub>it<sub>a</sub>ti<sub>on o</sub>f <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>me</sub>t<sub>r</sub>i<sub>cs use</sub>d t<sub>o va</sub>lid<sub>a</sub>t<sub>e</sub> bi<sub>nary c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s.</sub> C<sub>urren</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s o</sub>f <sub>mo</sub>d<sub>e</sub>l <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>re</sub>l<sub>y on s</sub>i<sub>ng</sub>l<sub>e per</sub>f<sub>ormance me</sub>t<sub>r</sub>i<sub>cs an</sub>d d<sub>o no</sub>t <sub>prov</sub>id<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on on w</sub>h<sub>ere</sub> i<sub>n</sub> th<sub>e covar</sub>i<sub>a</sub>t<sub>e space</sub> th<sub>e mo</sub>d<sub>e</sub>l i<sub>s</sub> <sub>un</sub>d<sub>erper</sub>f<sub>orm</sub>i<sub>ng.</sub> LAUC <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es a</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l<sub>-</sub>l<sub>eve</sub>l <sub>per</sub>f<sub>ormance assessmen</sub>t <sub>an</sub>d i<sub>s use</sub>d i<sub>n our</sub> <sub>propose</sub>d <sub>nove</sub>l LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m.</sub> Th<sub>e</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> id<sub>en</sub>tifi<sub>es</sub> <sub>reg</sub>i<sub>ons</sub> i<sub>n</sub> th<sub>e</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> <sub>space</sub> <sub>w</sub>ith hi<sub>g</sub>h <sub>or</sub> l<sub>ow mo</sub>d<sub>e</sub>l <sub>per</sub>f<sub>ormance.</sub> B<sub>y us</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d LAUC f<sub>unc</sub>ti<sub>ona</sub>l<sub>,</sub> LOCATOR <sub>avo</sub>id<sub>s</sub> d<sub>a</sub>t<sub>a su</sub>b<sub>se</sub>tti<sub>ng an</sub>d l<sub>oss</sub> <sub>o</sub>f <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>power.</sub> W<sub>e</sub> <sub>prov</sub>id<sub>e</sub>d i<sub>n</sub>t<sub>u</sub>iti<sub>on</sub> b<sub>e</sub>hi<sub>n</sub>d th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d LAUC f<sub>unc</sub>ti<sub>ona</sub>l b<sub>y</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t it i<sub>s</sub> <sub>a</sub>l<sub>ge</sub>b<sub>ra</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>equa</sub>l t<sub>o</sub> th<sub>e</sub> <sub>ra</sub>ti<sub>o</sub> <sub>o</sub>f <sub>covar</sub>i<sub>ance</sub> <sub>an</sub>d <sub>expec</sub>t<sub>a</sub>ti<sub>on.</sub> W<sub>e</sub> <sub>a</sub>l<sub>so</sub> <sub>prov</sub>id<sub>e</sub>d <sub>assump</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>guaran</sub>t<sub>y</sub> th<sub>a</sub>t<sub>,</sub> <sub>on</sub> <sub>average,</sub> th<sub>e</sub> localized LAUC functional has the right ordering. Using non-parametric two-sample U-statistic as<sub>y</sub>mptotic theor<sub>y</sub>, <sub>we</sub> d<sub>er</sub>i<sub>ve</sub>d <sub>c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm</sub> <sub>p</sub>l<sub>ug-</sub>i<sub>n</sub> <sub>var</sub>i<sub>ance</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>ors</sub> <sub>an</sub>d l<sub>og</sub>it<sub>-s</sub>t<sub>a</sub>bili<sub>ze</sub>d <sub>con</sub>fid<sub>ence</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s</sub> th<sub>a</sub>t <sub>e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>e</sub> th<sub>e</sub> <sub>nee</sub>d f<sub>or compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y</sub> i<sub>n</sub>t<sub>ens</sub>i<sub>ve resamp</sub>li<sub>ng</sub> i<sub>n</sub> l<sub>arge c</sub>li<sub>n</sub>i<sub>ca</sub>l <sub>co</sub>h<sub>or</sub>t<sub>s.</sub> B<sub>eyon</sub>d id<sub>en</sub>tif<sub>y</sub>i<sub>ng per</sub>f<sub>ormance</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y,</sub> LOCATOR <sub>prov</sub>id<sub>es</sub> <sub>a</sub> di<sub>agnos</sub>ti<sub>c</sub> t<sub>axonomy</sub> t<sub>o</sub> di<sub>sen</sub>t<sub>ang</sub>l<sub>e</sub> th<sub>e</sub> <sub>causes</sub> <sub>o</sub>f <sub>un</sub>d<sub>erper</sub>f<sub>ormance.</sub> Th<sub>roug</sub>h <sub>our</sub> t<sub>wo</sub> axis diagnostic matrix and a formal F-test for flat predicted risk, researchers can distinguish genuine structura <sub>mo</sub>d<sub>e</sub>l <sub>m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on</sub> f<sub>rom</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>co</sub>h<sub>or</sub>t <sub>case-m</sub>i<sub>x</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ons.</sub> Thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on</sub> <sub>may</sub> <sub>preven</sub>t <sub>unnecessary</sub> <sub>mo</sub>d<sub>e</sub>l <sub>re</sub>fitti<sub>ng</sub> <sub>w</sub>h<sub>en</sub> <sub>un</sub>d<sub>erper</sub>f<sub>ormance</sub> i<sub>s</sub> d<sub>r</sub>i<sub>ven</sub> b<sub>y</sub> <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> <sub>co</sub>h<sub>or</sub>t h<sub>omogene</sub>it<sub>y,</sub> <sub>w</sub>hil<sub>e</sub> <sub>p</sub>i<sub>npo</sub>i<sub>n</sub>ti<sub>ng</sub> <sub>spec</sub>ifi<sub>c</sub> <sub>p</sub>h<sub>eno</sub>t<sub>yp</sub>i<sub>c</sub> <sub>su</sub>b<sub>groups</sub> <sub>w</sub>h<sub>ere</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>mo</sub>d<sub>e</sub>l <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>or</sub> <sub>nove</sub>l bi<sub>omar</sub>k<sub>er</sub> di<sub>scovery</sub> <sub>are</sub> <sub>requ</sub>i<sub>re</sub>d<sub>.</sub> A<sub>pp</sub>li<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> <sub>gu</sub>id<sub>e</sub>li<sub>nes-recommen</sub>d<sub>e</sub>d AHA PREVENT <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> LOCATOR id<sub>en</sub>tifi<sub>e</sub>d it<sub>s</sub> hi<sub>g</sub>h <sub>an</sub>d l<sub>ow per</sub>f<sub>ormance reg</sub>i<sub>ons</sub> i<sub>n</sub> NHANES data (N = 17, 748; AUC = 0.88 in high-LOCATOR vs. 0.73 in low-competence regions). The same hi<sub>g</sub>h <sub>an</sub>d l<sub>ow per</sub>f<sub>ormance reg</sub>i<sub>ons were success</sub>f<sub>u</sub>ll<sub>y va</sub>lid<sub>a</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e prospec</sub>ti<sub>ve</sub> UK Bi<sub>o</sub>b<sub>an</sub>k <sub>co</sub>h<sub>or</sub>t<sub>.</sub> SHAP i<sub>n</sub>fl<sub>uence</sub> d<sub>ecompos</sub>iti<sub>on</sub> id<sub>en</sub>tifi<sub>e</sub>d <sub>o</sub>ld<sub>er</sub> <sub>age,</sub> <sub>an</sub>tih<sub>yper</sub>t<sub>ens</sub>i<sub>ve</sub> t<sub>rea</sub>t<sub>men</sub>t<sub>,</sub> <sub>s</sub>t<sub>a</sub>ti<sub>n</sub> th<sub>erapy,</sub> di<sub>a</sub>b<sub>e</sub>t<sub>es,</sub> <sub>an</sub>d <sub>smo</sub>ki<sub>ng</sub> <sub>as</sub> th<sub>e ma</sub>i<sub>n con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>ors</sub> t<sub>o</sub> th<sub>e</sub> l<sub>ower per</sub>f<sub>ormance o</sub>f th<sub>e</sub> PREVENT <sub>mo</sub>d<sub>e</sub>l<sub>.</sub>

## 7.1 Remediation Strategies for Low-Competence Regions

## 7.2 Strengths and Limitations

S<sub>evera</sub>l li<sub>m</sub>it<sub>a</sub>ti<sub>ons warran</sub>t <sub>cons</sub>id<sub>era</sub>ti<sub>on.</sub> Fi<sub>rs</sub>t<sub>,</sub> NHANES <sub>prov</sub>id<sub>es mor</sub>t<sub>a</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ow-up</sub> b<sub>u</sub>t l<sub>ac</sub>k<sub>s non-</sub>f<sub>a</sub>t<sub>a</sub>l <sub>car</sub>di<sub>ovascu</sub>l<sub>ar even</sub>t<sub>s, w</sub>hi<sub>c</sub>h <sub>are par</sub>t <sub>o</sub>f th<sub>e</sub> PREVENT <sub>compos</sub>it<sub>e ou</sub>t<sub>come.</sub> Thi<sub>s</sub> di<sub>scor</sub>d<sub>ance may a</sub>tt<sub>enua</sub>t<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>per</sub>f<sub>ormance an</sub>d li<sub>m</sub>it <sub>compara</sub>bilit<sub>y w</sub>ith <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>va</sub>lid<sub>a</sub>ti<sub>on s</sub>t<sub>u</sub>di<sub>es.</sub> S<sub>econ</sub>d<sub>,</sub> NHANES i<sub>s cross-</sub> sectional, precluding the assessment of dynamic risk factor trajectories that could influence model performance <sub>over</sub> ti<sub>me.</sub> Thi<sub>r</sub>d<sub>,</sub> <sub>our</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> i<sub>s</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> U<sub>.</sub>S<sub>.</sub> <sub>popu</sub>l<sub>a</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>genera</sub>li<sub>za</sub>bilit<sub>y</sub> t<sub>o</sub> <sub>o</sub>th<sub>er</sub> <sub>se</sub>tti<sub>ngs</sub> <sub>rema</sub>i<sub>ns</sub> untested. Fourth, the fixed ex<sub>p</sub>onential <sub>p</sub>arameter (ex<sub>p</sub>=15) for similarit<sub>y</sub> wei<sub>g</sub>htin<sub>g</sub>, while validated in <sub>p</sub>rior work[7], ma<sub>y</sub> not be o<sub>p</sub>timal for all datasets. Fifth, SHAP associations are not causal; the<sub>y</sub> identif<sub>y</sub> correlates of <sub>per</sub>f<sub>ormance</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> b<sub>u</sub>t d<sub>o</sub> <sub>no</sub>t i<sub>mp</sub>l<sub>y</sub> <sub>causa</sub>l d<sub>e</sub>t<sub>erm</sub>i<sub>nan</sub>t<sub>s.</sub>

## 7.3 Extensions and Future Directions

F<sub>u</sub>t<sub>ure</sub> <sub>researc</sub>h <sub>s</sub>h<sub>ou</sub>ld <sub>exp</sub>l<sub>ore</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> <sub>we</sub>i<sub>g</sub>hti<sub>ng</sub> <sub>sc</sub>h<sub>emes</sub> t<sub>o</sub> <sub>op</sub>ti<sub>m</sub>i<sub>ze</sub> th<sub>e</sub> <sub>exponen</sub>ti<sub>a</sub>l <sub>parame</sub>t<sub>er</sub> f<sub>or</sub> <sub>spec</sub>ifi<sub>c</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s,</sub> <sub>po</sub>t<sub>en</sub>ti<sub>a</sub>ll<sub>y</sub> b<sub>y</sub> <sub>max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> <sub>su</sub>b<sub>group</sub> <sub>separa</sub>ti<sub>on</sub> <sub>or</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d AUC <sub>es</sub>ti<sub>ma</sub>t<sub>es</sub> <sub>across</sub> b<sub>oo</sub>t<sub>-</sub> stra<sub>p</sub> sam<sub>p</sub>les. Extension to other risk models (e.<sub>g</sub>., PCE, SCORE2) would reveal whether observed hetero<sub>g</sub>eneit<sub>y</sub> i<sub>s mo</sub>d<sub>e</sub>l<sub>-spec</sub>ifi<sub>c or sys</sub>t<sub>ema</sub>ti<sub>c.</sub> I<sub>ncorpora</sub>ti<sub>ng</sub> ti<sub>me-</sub>t<sub>o-even</sub>t <sub>ou</sub>t<sub>comes an</sub>d ti<sub>me-</sub>d<sub>epen</sub>d<sub>en</sub>t AUC <sub>me</sub>t<sub>r</sub>i<sub>cs wou</sub>ld b<sub>e</sub>tt<sub>er a</sub>li<sub>gn w</sub>ith th<sub>e surv</sub>i<sub>va</sub>l f<sub>ramewor</sub>k <sub>o</sub>f PREVENT<sub>.</sub> E<sub>x</sub>t<sub>erna</sub>l <sub>va</sub>lid<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> di<sub>verse co</sub>h<sub>or</sub>t<sub>s-suc</sub>h <sub>as</sub> UK Bi<sub>o</sub>b<sub>an</sub>k <sub>or</sub> MESA<sub>,</sub> i<sub>s</sub> <sub>essen</sub>ti<sub>a</sub>l t<sub>o</sub> <sub>assess</sub> <sub>genera</sub>li<sub>za</sub>bilit<sub>y</sub> <sub>an</sub>d <sub>exp</sub>l<sub>ore</sub> <sub>rac</sub>i<sub>a</sub>l/<sub>e</sub>th<sub>n</sub>i<sub>c</sub> <sub>var</sub>i<sub>a</sub>ti<sub>on.</sub> Fi<sub>na</sub>ll<sub>y,</sub> <sub>prospec</sub>ti<sub>ve</sub> i<sub>mp</sub>l<sub>emen-</sub> t<sub>a</sub>ti<sub>on s</sub>t<sub>u</sub>di<sub>es cou</sub>ld <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e c</sub>li<sub>n</sub>i<sub>ca</sub>l <sub>u</sub>tilit<sub>y o</sub>f fl<sub>agg</sub>i<sub>ng</sub> l<sub>ow-per</sub>f<sub>ormance su</sub>b<sub>groups</sub> t<sub>o gu</sub>id<sub>e</sub> t<sub>arge</sub>t<sub>e</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>reca</sub>lib<sub>ra</sub>ti<sub>on or supp</sub>l<sub>emen</sub>t<sub>ary r</sub>i<sub>s</sub>k <sub>assessmen</sub>t<sub>.</sub>

I<sub>n</sub> <sub>a</sub>dditi<sub>on,</sub> <sub>a</sub> <sub>compar</sub>i<sub>son</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> di<sub>v</sub>i<sub>s</sub>i<sub>on</sub> <sub>me</sub>th<sub>o</sub>d <sub>o</sub>f XGB<sub>oos</sub>t <sub>an</sub>d b<sub>y</sub> <sub>us</sub>i<sub>ng</sub> SHAPl<sub>ey</sub> <sub>va</sub>l<sub>ues</sub> t<sub>o</sub> di<sub>v</sub>id<sub>e</sub> hi<sub>g</sub>h<sub>- an</sub>d l<sub>ow-per</sub>f<sub>ormance su</sub>b<sub>groups can a</sub>l<sub>so</sub> b<sub>e</sub> t<sub>es</sub>t<sub>e</sub>d<sub>,</sub> i<sub>n or</sub>d<sub>er</sub> t<sub>o ver</sub>if<sub>y</sub> th<sub>e a</sub>li<sub>gnmen</sub>t <sub>o</sub>f di<sub>rec</sub>t <sub>com</sub>bi<sub>na</sub>ti<sub>on</sub> <sub>o</sub>f <sub>su</sub>b<sub>group</sub> <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> <sub>va</sub>l<sub>ues</sub> <sub>an</sub>d XGB<sub>oos</sub>t <sub>me</sub>th<sub>o</sub>d <sub>an</sub>d <sub>ver</sub>if<sub>y</sub> th<sub>e</sub> XGB<sub>oos</sub>t <sub>me</sub>th<sub>o</sub>d <sub>a</sub>bilit<sub>y</sub> i<sub>n</sub> id<sub>en</sub>tif<sub>y</sub>i<sub>ng</sub> <sub>mu</sub>lti <sub>var</sub>i<sub>an</sub>t i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>ng</sub> l<sub>oca</sub>l <sub>per</sub>f<sub>ormance</sub> <sub>scores.</sub> A<sub>no</sub>th<sub>er</sub> <sub>compar</sub>i<sub>son</sub> b<sub>e</sub>t<sub>ween</sub> <sub>our</sub> SHAP <sub>summary</sub> <sub>p</sub>lot and direct SHAP summar<sub>y</sub> <sub>p</sub>lot derived directl<sub>y</sub> from the existin<sub>g</sub> model (PREVENT) can also be <sub>p</sub>otentiall<sub>y</sub> <sub>mean</sub>i<sub>ng</sub>f<sub>u</sub>l<sub>.</sub>

## 7.4 Conclusions

Th<sub>e</sub> <sub>nove</sub>l LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>an</sub>d l<sub>oca</sub>li<sub>ze</sub>d LAUC f<sub>unc</sub>ti<sub>ona</sub>l <sub>propose</sub>d i<sub>n</sub> thi<sub>s</sub> <sub>wor</sub>k <sub>can</sub> b<sub>e</sub> <sub>use</sub>d t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> l<sub>ow</sub> <sub>an</sub>d hi<sub>g</sub>h <sub>per</sub>f<sub>ormance reg</sub>i<sub>ons o</sub>f bi<sub>nary c</sub>l<sub>ass</sub>ifi<sub>ers.</sub> A<sub>s</sub> bi<sub>nary c</sub>l<sub>ass</sub>ifi<sub>ers con</sub>ti<sub>nue</sub> t<sub>o</sub> i<sub>n</sub>f<sub>orm</sub> hi<sub>g</sub>h<sub>-s</sub>t<sub>a</sub>k<sub>es</sub> h<sub>ea</sub>lth<sub>care</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> f<sub>or</sub> <sub>m</sub>illi<sub>ons</sub> <sub>o</sub>f <sub>pa</sub>ti<sub>en</sub>t<sub>s,</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>s</sub>hift<sub>s</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>para</sub>di<sub>gm</sub> f<sub>rom</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e,</sub> <sub>poo</sub>l<sub>e</sub>d <sub>co</sub>h<sub>or</sub>t <sub>summary s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs</sub> t<sub>owar</sub>d l<sub>oca</sub>li<sub>ze</sub>d <sub>per</sub>f<sub>ormance me</sub>t<sub>r</sub>i<sub>cs.</sub> W<sub>e are wor</sub>ki<sub>ng on genera</sub>li<sub>z</sub>i<sub>ng</sub> thi<sub>s approac</sub>h b<sub>eyon</sub>d th<sub>e c</sub>li<sub>n</sub>i<sub>ca</sub>l d<sub>oma</sub>i<sub>n an</sub>d <sub>r</sub>i<sub>s</sub>k <sub>pre</sub>di<sub>c</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s.</sub>

## 8 Acknowledgements

A<sub>nna</sub> G<sub>yor</sub>f<sub>y-</sub>K<sub>ere</sub>k<sub>es wou</sub>ld lik<sub>e</sub> t<sub>o</sub> th<sub>an</sub>k th<sub>e</sub> ETH C<sub>en</sub>t<sub>er</sub> f<sub>or</sub> L<sub>earn</sub>i<sub>ng</sub> S<sub>ys</sub>t<sub>ems</sub> f<sub>or</sub> th<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> t<sub>o</sub> h<sub>er</sub>˝ PhD<sub>.</sub> Th<sub>e au</sub>th<sub>ors gra</sub>t<sub>e</sub>f<sub>u</sub>ll<sub>y ac</sub>k<sub>now</sub>l<sub>e</sub>d<sub>ge</sub> i<sub>ns</sub>tit<sub>u</sub>ti<sub>ona</sub>l <sub>suppor</sub>t f<sub>rom</sub> ETH Z<sub>ur</sub>i<sub>c</sub>h<sub>,</sub> S<sub>w</sub>it<sub>zer</sub>l<sub>an</sub>d<sub>.</sub> Thi<sub>s researc</sub>h <sub>was</sub> <sub>con</sub>d<sub>uc</sub>t<sub>e</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> UK Bi<sub>o</sub>b<sub>an</sub>k R<sub>esource un</sub>d<sub>er app</sub>li<sub>ca</sub>ti<sub>on num</sub>b<sub>ers</sub> 81959 <sub>an</sub>d 206575<sub>.</sub> W<sub>e ex</sub>t<sub>en</sub>d <sub>our s</sub>i<sub>ncere</sub> <sub>gra</sub>tit<sub>u</sub>d<sub>e</sub> t<sub>o</sub> <sub>a</sub>ll <sub>par</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s</sub> <sub>an</sub>d <sub>s</sub>t<sub>u</sub>d<sub>y</sub> <sub>organ</sub>i<sub>zers</sub> <sub>o</sub>f NHANES <sub>an</sub>d UK Bi<sub>o</sub>b<sub>an</sub>k f<sub>or</sub> <sub>ma</sub>ki<sub>ng</sub> th<sub>ese</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e.</sub>

## 8.1 Statement on AI use

I<sub>n</sub> th<sub>e prepara</sub>ti<sub>on o</sub>f thi<sub>s wor</sub>k<sub>, genera</sub>ti<sub>ve</sub> AI t<sub>oo</sub>l<sub>s were use</sub>d t<sub>o ass</sub>i<sub>s</sub>t <sub>w</sub>ith <sub>co</sub>d<sub>e wr</sub>iti<sub>ng.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub> Dee<sub>p</sub>Seek (https://www.deepseek.com/) and Gemini (https://gemini.google.com/) were used to assist with P<sub>y</sub>thon code writin<sub>g</sub>. Claude Deskto<sub>p</sub> (free <sub>p</sub>lan, https://claude.ai/download) was used to assist <sub>w</sub>ith R l<sub>anguage co</sub>d<sub>e wr</sub>iti<sub>ng.</sub> Th<sub>ese</sub> t<sub>oo</sub>l<sub>s were use</sub>d f<sub>or co</sub>d<sub>e-re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>ass</sub>i<sub>s</sub>t<sub>ance, suc</sub>h <sub>as sugges</sub>ti<sub>ng co</sub>d<sub>e sn</sub>i<sub>ppe</sub>t<sub>s,</sub> d<sub>e</sub>b<sub>ugg</sub>i<sub>ng,</sub> <sub>an</sub>d <sub>exp</sub>l<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> <sub>error</sub> <sub>messages.</sub> All AI<sub>-ass</sub>i<sub>s</sub>t<sub>e</sub>d <sub>co</sub>d<sub>e</sub> <sub>was</sub> <sub>rev</sub>i<sub>ewe</sub>d<sub>,</sub> t<sub>es</sub>t<sub>e</sub>d<sub>,</sub> <sub>an</sub>d <sub>a</sub>d<sub>ap</sub>t<sub>e</sub>d b<sub>y</sub> <sub>au</sub>th<sub>ors,</sub> <sub>w</sub>h<sub>o</sub> t<sub>a</sub>k<sub>e</sub> f<sub>u</sub>ll <sub>respons</sub>ibilit<sub>y</sub> f<sub>or</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>co</sub>d<sub>e</sub> <sub>an</sub>d <sub>ana</sub>l<sub>yses.</sub>

D<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> <sub>prepara</sub>ti<sub>on</sub> <sub>o</sub>f thi<sub>s</sub> <sub>manuscr</sub>i<sub>p</sub>t<sub>,</sub> l<sub>arge</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s,</sub> <sub>spec</sub>ifi<sub>ca</sub>ll<sub>y</sub> G<sub>em</sub>i<sub>n</sub>i P<sub>ro</sub> <sub>an</sub>d Cl<sub>au</sub>d<sub>e</sub> P<sub>ro</sub> <sub>were</sub> <sub>emp</sub>l<sub>oye</sub>d t<sub>o</sub> <sub>ass</sub>i<sub>s</sub>t <sub>w</sub>ith L<sub>a</sub>T<sub>e</sub>X d<sub>e</sub>b<sub>ugg</sub>i<sub>ng,</sub> <sub>ver</sub>if<sub>y</sub> th<sub>e</sub> <sub>correc</sub>t<sub>ness</sub> <sub>o</sub>f <sub>se</sub>l<sub>ec</sub>t <sub>proo</sub>f<sub>s,</sub> <sub>an</sub>d <sub>re</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>aca</sub>d<sub>em</sub>i<sub>c</sub> <sub>wr</sub>iti<sub>ng.</sub> Th<sub>e</sub> l<sub>oca</sub>li<sub>ze</sub>d AUC f<sub>unc</sub>ti<sub>ona</sub>l d<sub>e</sub>fi<sub>n</sub>iti<sub>on,</sub> th<sub>e</sub> LOCATOR <sub>a</sub>l<sub>gor</sub>ith<sub>m, an</sub>d P<sub>ropos</sub>iti<sub>ons</sub> 1<sub>-</sub>3 <sub>w</sub>ith <sub>proo</sub>f<sub>s are en</sub>ti<sub>re</sub>l<sub>y</sub> due to the authors: no use of AI technolo<sub>g</sub>ies was involved. A conversation with Gemini 3.6 Flash (Goo<sub>g</sub>le) <sub>was use</sub>d t<sub>o genera</sub>t<sub>e</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l d<sub>ra</sub>ft <sub>o</sub>f th<sub>e proo</sub>f<sub>s o</sub>f P<sub>ropos</sub>iti<sub>ons</sub> 4<sub>-</sub>6<sub>, w</sub>hi<sub>c</sub>h <sub>were su</sub>b<sub>sequen</sub>tl<sub>y rewor</sub>k<sub>e</sub>d b<sub>y</sub> the authors. Text was scanned for errors with the assistance of Gemini 3.6 Flash (Goo<sub>g</sub>le). Authors take full <sub>respons</sub>ibilit<sub>y</sub> f<sub>or</sub> th<sub>e</sub> <sub>correc</sub>t<sub>ness</sub> <sub>o</sub>f th<sub>e</sub> <sub>en</sub>ti<sub>re</sub> <sub>con</sub>t<sub>en</sub>t <sub>o</sub>f thi<sub>s</sub> <sub>manuscr</sub>i<sub>p</sub>t<sub>.</sub>

# SUPPLEMENTARY MATERIAL

## 9 Proofs from Section 2

Let the covariate s<sub>p</sub>ace X . Let $f : \mathcal { X } $ R b<sub>e</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> W<sub>e</sub> <sub>a</sub>l<sub>so</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
F ^ { + } ( t ) = \mathbb { P } ( f ( X ^ { + } ) \leq t ) ,\tag{11}
$$

$$
F ^ { - } ( t ) = \mathbb { P } ( f ( X ^ { - } ) \leq t ) ,\tag{12}
$$

(13)

L<sub>e</sub>t <sub>us rem</sub>i<sub>n</sub>d <sub>ourse</sub>l<sub>ves,</sub> th<sub>a</sub>t th<sub>e we</sub>i<sub>g</sub>ht<sub>e</sub>d AUC i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 1<sub>.</sub>

Proposition 1 (Under Uniform Weights the Localized AUC Reduces to the AUC). Let X denote the covariate space. If we sample a patient with an event and covariates $X ^ { + } \sim F ^ { + }$ , where $F ^ { + }$ is the sampling distribution of our event patients, and a patient with no event with covariates $X ^ { - } \sim F ^ { - }$ . Let $x _ { i } \in { \mathcal { X } }$ be covariates for our index patient i. We also have a risk prediction model $f : \mathcal { X } \to [ 0 , 1 ]$ . In addition, we also have a weight function $w : \mathcal { X } \times \mathcal { X }  [ 0 , 1 ]$ , that is some constant c. Then the localized AUC is equivalent to the traditional AUC:

$$
\mathrm { L A U C } ( x _ { i } ) = \mathrm { A U C } .\tag{4}
$$

Proof.

$$
\begin{array} { r l } & { \mathrm { L A U C } ( x _ { i } ) = \frac { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( x _ { i } , X ^ { + } ) w ( x _ { i } , X ^ { - } ) \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) \right] } { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( x _ { i } , X ^ { + } ) w ( x _ { i } , X ^ { - } ) \right] } } \\ & { \qquad = \frac { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ c ^ { 2 } \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) \right] } { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ c ^ { 2 } \right] } } \\ & { \qquad = \mathrm { A U C } . } \end{array}\tag{14}
$$

(15)

(16)

If <sub>we</sub> l<sub>oo</sub>k <sub>a</sub>t th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>or,</sub> <sub>we</sub> <sub>ge</sub>t

$$
\mathrm { I . A U T C } ( x _ { \lambda } ) = \frac { \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } w ( x _ { i } , x _ { k } ^ { + } ) w ( x _ { i } , x _ { l } ^ { - } ) \mathbb { 1 } \left( f ( x _ { k } ^ { + } ) > f ( x _ { l } ^ { - } ) \right) } { \cdots }\tag{17}
$$

$$
- \textstyle \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } c ^ { 2 } \mathbb { 1 } \left( f ( x _ { k } ^ { + } ) > f ( x _ { l } ^ { - } ) \right)
$$

$$
\textstyle \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } c ^ { 2 }\tag{18}
$$

$$
= \frac { c ^ { 2 } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } \mathbb { 1 } \left( f ( x _ { k } ^ { + } ) > f ( x _ { l } ^ { - } ) \right) } { K L c ^ { 2 } }
$$

$$
= \mathrm { A U C } .\tag{19}
$$

(20)

Proposition 2 (Localized AUC as Covariance). Suppose number of events is n and number of non-events is m. Define $W _ { k l } = w _ { k } ^ { + } w _ { l } ^ { - }$ where $w ^ { + }$ is the weight vector of event participants, and w<sup>−</sup> is the weight vector of non-event participants. Then $W \in R ^ { n \times m }$ With two types of weight vectors $w _ { i } , w _ { j }$ , The $W ^ { ( i ) } , W ^ { ( j ) }$ can be calculated. For event participants k, a and non-event participants l, b, define $\begin{array} { r } { I _ { k l } = \mathbb { 1 } \big ( f \big ( x _ { k } \big ) > f \big ( x _ { l } \big ) \big ) + \frac { 1 } { 2 } \mathbb { 1 } \big ( f \big ( x _ { k } \big ) = f \big ( x _ { l } \big ) \big ) \in R ^ { n \times m } } \end{array}$ $\begin{array} { r } { \gamma _ { k l } = \frac { W _ { k l } ^ { ( i ) } } { W _ { k l } ^ { ( j ) } } } \end{array}$ and $\begin{array} { r } { p _ { k l } = \frac { W _ { k l } ^ { ( j ) } } { \sum _ { a , b } W _ { a b } ^ { ( j ) } } } \end{array}$ . Ifwe think ofp as aprobability distribution ofk, l, then $\mathrm { L A U C } ( x _ { i } ) - \mathrm { L A U C } ( x _ { j } ) =$ $\frac { C o \nu _ { p } ( \gamma , I ) } { \mathbb { E } _ { 1 } ( \gamma ) }$

Proof. We know

$$
\mathbb { E } _ { p } ( \gamma ) = \sum _ { k , l } p _ { k , l } \gamma _ { k , l }\tag{21}
$$

$$
= \sum _ { k , l } \left( \frac { W _ { k l } ^ { ( i ) } } { W _ { k , l } ^ { ( j ) } } \right) \left( \frac { W _ { k l } ^ { ( j ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } } \right)\tag{22}
$$

$$
= \sum _ { k , l } \left( \frac { W _ { k l } ^ { ( i ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } } \right)\tag{23}
$$

$$
= \frac { \sum _ { k , l } W _ { k l } ^ { ( i ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } }\tag{24}
$$

<sub>an</sub>d

$$
\mathbb { E } _ { p } ( I ) = \sum _ { k , l } p _ { k , l } I _ { k , l }
$$

$$
= \sum _ { k , l } I _ { k , l } \left( \frac { W _ { k l } ^ { ( j ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } } \right)\tag{25}
$$

(26)

$$
= { \mathrm { L A U C } } ( x _ { j } )\tag{27}
$$

<sub>an</sub>d

$$
\begin{array} { l } { \mathbb { E } _ { p } ( \gamma I ) = \displaystyle \sum _ { k , l } p _ { k , l } \gamma _ { k , l } I _ { k , l } } \\ { = \displaystyle \sum _ { k , l } I _ { k , l } \left( \frac { W _ { k l } ^ { ( i ) } } { W _ { k , l } ^ { ( j ) } } \right) \left( \frac { W _ { k l } ^ { ( j ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } } \right) } \\ { = \displaystyle \sum _ { k , l } I _ { k , l } \left( \frac { W _ { k l } ^ { ( i ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } } \right) } \\ { = \displaystyle \sum _ { k , l } I _ { k , l } W _ { a l } ^ { ( i ) } . } \end{array}\tag{28}
$$

(29)

(30)

(31)

Th<sub>en</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } { \frac { \mathrm { C o v } _ { p } ( \gamma , I ) } { \mathbb { E } _ { p } ( \gamma ) } = \frac { \mathbb { E } _ { p } ( \gamma I ) - \mathbb { E } _ { p } ( \gamma ) \mathbb { E } _ { p } ( I ) } { \mathbb { E } _ { p } ( \gamma ) } } & { } \\ & { = \frac { \sum _ { k , l } F _ { k , l } W _ { k l } ^ { ( i ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } } - \mathrm { L A U C } ( x _ { j } ) \frac { \sum _ { k , l } W _ { k l } ^ { ( i ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } } } \\ & { = \frac { \sum _ { k , l } W _ { k l } ^ { ( i ) } } { \sum _ { a , b } W _ { a , b } ^ { ( j ) } } } \\ & { = \frac { \sum _ { k , l } I _ { k , l } W _ { k l } ^ { ( i ) } - \mathrm { L A U C } ( x _ { j } ) \sum _ { k , l } W _ { k l } ^ { ( i ) } } { \sum _ { k , l } W _ { k l } ^ { ( i ) } } } \end{array}\tag{32}
$$

(33)

(34)

$$
= \frac { \sum _ { k , l } I _ { k , l } W _ { k l } ^ { ( i ) } } { \sum _ { k , l } W _ { k l } ^ { ( i ) } } - \mathrm { L A U C } ( x _ { j } )\tag{35}
$$

$$
= { \mathrm { L A U C } } ( x _ { i } ) - { \mathrm { L A U C } } ( x _ { j } ) .\tag{36}
$$

L<sub>e</sub>t th<sub>e</sub> <sub>covar</sub>i<sub>a</sub>t<sub>e</sub> <sub>space</sub> $\mathcal { X }$ be partitioned into two disjoint, phenotypically distinct regions $S _ { 1 }$ <sub>an</sub>d $S _ { 2 }$ <sub>.</sub> We a<sub>ssu</sub>me

$S _ { 1 } \cap S _ { 2 } = \emptyset$ <sub>an</sub>d $S _ { 1 } \cup S _ { 2 } = \mathcal { X }$ <sub>.</sub> L<sub>e</sub>t <sub>us a</sub>l<sub>so</sub> d<sub>e</sub>fi<sub>ne</sub> $( S _ { 1 } \times S _ { 1 } ) ^ { C }$ as $\mathcal { X } \times \mathcal { X } \backslash S _ { 1 } \times S _ { 1 }$ <sub>an</sub>d $( S _ { 2 } \times S _ { 2 } ) ^ { C }$ as $\mathcal { X } \times \mathcal { X } \backslash S _ { 2 } \times S _ { 2 }$ W<sub>e a</sub>l<sub>so</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
F _ { i } ^ { + } ( t ) = \mathbb { P } ( f ( X ^ { + } ) \leq t \mid X ^ { + } , X ^ { - } \in S _ { i } ) \quad { \mathrm { ~ f o r ~ } } i \in \{ 1 , 2 \} { \mathrm { ~ a n d ~ } }\tag{37}
$$

$$
F _ { i } ^ { - } ( t ) = \mathbb { P } ( f ( X ^ { - } ) \leq t \mid X ^ { + } , X ^ { - } \in S _ { i } ) \quad { \mathrm { ~ f o r ~ } } i \in \{ 1 , 2 \} .\tag{38}
$$

D<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>popu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>n</sub>t<sub>ra-reg</sub>i<sub>on</sub> <sub>an</sub>d <sub>cross-</sub>b<sub>oun</sub>d<sub>ary</sub> <sub>concor</sub>d<sub>ance</sub> <sub>parame</sub>t<sub>ers:</sub>

$$
C _ { 1 } = \mathbb { P } _ { F ^ { + } \times F ^ { - } } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \in S _ { 1 } ) = \int _ { - \infty } ^ { \infty } F _ { 1 } ^ { - } ( t ) d F _ { 1 } ^ { + } ( t )
$$

$$
C _ { 2 } = \mathbb { P } _ { F ^ { + } \times F ^ { - } } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \in S _ { 2 } ) = \int _ { - \infty } ^ { \infty } F _ { 2 } ^ { - } ( t ) d F _ { 2 } ^ { + } ( t )
$$

$$
C _ { 1 , 2 } = \mathbb { P } _ { F ^ { + } \times F ^ { - } } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } \in S _ { 1 } , X ^ { - } \in S _ { 2 } ) = \int _ { - \infty } ^ { \infty } F _ { 2 } ^ { - } ( t ) d F _ { 1 } ^ { + } ( t )
$$

$$
C _ { 2 , 1 } = \mathbb { P } _ { F ^ { + } \times F ^ { - } } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } \in S _ { 2 } , X ^ { - } \in S _ { 1 } ) = \int _ { - \infty } ^ { \infty } F _ { 1 } ^ { - } ( t ) d F _ { 2 } ^ { + } ( t )
$$

$$
C _ { 1 } ^ { \mathrm { c r o s s } } = \mathbb { P } _ { F ^ { + } \times F ^ { - } } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid ( X ^ { + } , X ^ { - } ) \notin S _ { 1 } \times S _ { 1 } )
$$

$$
C _ { 2 } ^ { \mathrm { c r o s s } } = \mathbb { P } _ { F ^ { + } \times F ^ { - } } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid ( X ^ { + } , X ^ { - } ) \notin S _ { 2 } \times S _ { 2 } )
$$

## 9.1 Assumptions

Assumptions 1 (First-Order Stochastic Dominance). The reference model risk score distributions satisfy strict First-Order Stochastic Dominance (FSD) between regions $S _ { 1 }$ and $S _ { 2 }$ :

$$
\begin{array} { r l } & { \bullet \ C a s e s \ ( Y = 1 ) : } \\ & { \ f ( X _ { 1 } ^ { + } ) \xrightarrow [ s t ] { \geq } f ( X _ { 2 } ^ { + } ) \iff F _ { 1 } ^ { + } ( t ) \leq F _ { 2 } ^ { + } ( t ) \quad \forall t \in \mathbb { R } , X _ { 1 } ^ { + } \sim F ^ { + } ( \cdot \mid S _ { 1 } ) , X _ { 2 } ^ { + } \sim F ^ { + } ( \cdot \mid S _ { 2 } ) } \end{array}
$$

• Controls $( Y = 0 )$ :

$$
f ( X _ { 1 } ^ { - } ) \le f ( X _ { 2 } ^ { - } ) \iff F _ { 1 } ^ { - } ( t ) \ge F _ { 2 } ^ { - } ( t ) \quad \forall t \in \mathbb { R } , X _ { 1 } ^ { - } \sim F ^ { - } ( \cdot \mid S _ { 1 } ) , X _ { 2 } ^ { - } \sim F ^ { - } ( \cdot \mid S _ { 2 } )
$$

Assumptions 2 (The groups are seperated with high probability). Let $W _ { i } = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( X _ { i } , X ^ { + } ) w ( X _ { i } , X ^ { - } ) \right]$ and $W _ { j } = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( X _ { j } , X ^ { + } ) w ( X _ { j } , X ^ { - } ) \right]$

For $D _ { 1 } , D _ { 2 } , \epsilon , \eta \in ( 0 , 1 )$ , we have

$$
\mathbb { P } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left( \mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } \left[ \frac { w ( X _ { i } , X ^ { + } ) w ( X _ { i } , X ^ { - } ) } { W _ { i } } \right] > D _ { 1 } | ( X ^ { + } , X ^ { - } ) \in ( S _ { 1 } \times S _ { 1 } ) \right) > \epsilon \quad a n d ,\tag{39}
$$

$$
\mathbb { P } _ { X ^ { + } \sim F ^ { + } , X ^ { -- } \sim F ^ { - } } \left( \mathbb { E } _ { X _ { j } \sim P _ { X _ { j } \in S _ { 2 } } } \left[ \frac { w ( X _ { j } , X ^ { + } ) w ( X _ { j } , X ^ { - } ) } { W _ { j } } \right] < D _ { 2 } | ( X ^ { + } , X ^ { - } ) \in ( S _ { 2 } \times S _ { 2 } ) ^ { C } \right) > \epsilon \quad a n d ,\tag{40}
$$

$$
\mathbb { P } ( ( X ^ { + } , X ^ { - } ) \in S _ { 1 } \times S _ { 1 } ) > \frac { 1 } { 4 } - \eta \quad a n d ,\tag{41}
$$

$$
\mathbb { P } ( ( X ^ { + } , X ^ { - } ) \in ( S _ { 2 } \times S _ { 2 } ) ^ { C } ) < \frac { 1 } { 4 } + \eta \quad a n d ,\tag{42}
$$

$$
\mathbb { P } ( ( X ^ { + } , X ^ { - } ) \in S _ { 2 } \times S _ { 2 } ) < \frac { 1 } { 4 } + \eta \quad a n d ,\tag{43}
$$

$$
( \epsilon - ( 1 - C _ { 1 } ) ) D _ { 1 } \left( \frac { 1 } { 4 } - \eta \right) \ge ( C _ { 2 } + ( C _ { 2 } ^ { c r o s s } ( 1 - \epsilon ) + C _ { 2 } ^ { c r o s s } D _ { 2 } \epsilon ) \left( \frac { 1 } { 4 } + \eta \right) .\tag{44}
$$

## 9.2 Useful Lemma

Lemma 1.

$$
C _ { 2 } \leq C _ { 1 } ^ { c r o s s } \leq C _ { 1 } \quad a n d \quad C _ { 2 } \leq C _ { 2 } ^ { c r o s s } \leq C _ { 1 } .\tag{45}
$$

$$
C _ { 1 } ^ { c r o s s } = \mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \not \in S _ { 1 } \times S _ { 1 } )\tag{46}
$$

$$
= \mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } )\tag{47}
$$

$$
\mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) , X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } )
$$

$$
\mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } ) )\tag{48}
$$

$$
\mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) , X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) )
$$

$$
^ { - } \ \overline { { \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } ) ) } }\tag{49}
$$

$$
\mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) , X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 1 } ) )
$$

$$
+ \frac { \ J } { \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } ) ) }\tag{50}
$$

$$
\mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) , X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 2 } ) )
$$

$$
^ { + } \frac { \cdot } { \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } ) ) }\tag{51}
$$

$$
\mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) ) \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) )
$$

$$
\mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } ) )\tag{52}
$$

$$
\mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 1 } ) ) \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 1 } ) )
$$

$$
\mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } ) )\tag{53}
$$

$$
\begin{array} { r l } { { } _ { 1 } } & { { } \mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 2 } ) ) \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 2 } ) ) } \end{array}
$$

$$
\mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 2 } \times S _ { 2 } ) )\tag{54}
$$

$$
C _ { 1 , 2 } \mathbb { P } \big ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \big ) + C _ { 2 , 1 } \mathbb { P } \big ( X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 1 } ) \big ) + C _ { 2 } \mathbb { P } \big ( X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 2 } ) \big )\tag{55}
$$

$$
\mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) ) + \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 1 } ) ) + \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 2 } ) )
$$

$$
= \lambda _ { 1 2 } C _ { 1 , 2 } + \lambda _ { 2 1 } C _ { 2 , 1 } + \lambda _ { 2 2 } C _ { 2 } \mathrm { , a c o n v e x c o m b i n a t i o n o f { C s } . }\tag{56}
$$

(57)

<sub>w</sub>h<sub>ere</sub> $\lambda _ { 1 , 2 } = \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) ) , \lambda _ { 2 , 1 } \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 1 } ) )$ <sub>an</sub>d $\lambda _ { 2 } = \mathbb { P } ( X ^ { + } , X ^ { - } \in ( S _ { 2 } \times S _ { 2 } ) )$ <sub>an</sub>d $\lambda _ { 1 , 2 } , \lambda _ { 2 , 1 } , \lambda _ { 2 } \in [ 0 , 1 ]$

Si<sub>m</sub>il<sub>ar</sub>l<sub>y,</sub>

$$
C _ { 2 } ^ { c r o s s } = \mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \not \in S _ { 2 } \times S _ { 2 }\tag{58}
$$

$$
= \mathbb { P } ( f ( X ^ { + } ) > f ( X ^ { - } ) \mid X ^ { + } , X ^ { - } \in ( S _ { 1 } \times S _ { 2 } ) \cup ( S _ { 2 } \times S _ { 1 } ) \cup ( S _ { 1 } \times S _ { 1 } )\tag{59}
$$

$$
= \gamma _ { 1 2 } C _ { 1 , 2 } + \gamma _ { 2 1 } C _ { 2 , 1 } + \gamma _ { 1 1 } C _ { 1 } , \mathrm { a  c o n v e x  c o m b i n a t i o n o f } \mathrm { C s } .\tag{60}
$$

S<sub>o</sub> it i<sub>s enoug</sub>h t<sub>o s</sub>h<sub>ow,</sub> $C _ { 2 } \leq C _ { 1 , 2 } , C _ { 2 , 1 } \leq C _ { 1 }$

1. Lower Bound: $C _ { 1 , 2 } \geq C _ { 2 }$

B<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 1<sub>:</sub> $f ( X _ { 1 } ^ { + } ) \geq \underline { { { f } } } ( X _ { 2 } ^ { + } ) \iff F _ { 1 } ^ { + } ( t ) \leq F _ { 2 } ^ { + } ( t )$ , ∀t (cases in $S _ { 1 }$ <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>ca</sub>ll<sub>y</sub> d<sub>om</sub>i<sub>na</sub>t<sub>e</sub> <sub>cases</sub> i<sub>n</sub> $S _ { 2 } )$ B<sub>ecause</sub> $F _ { 2 } ^ { - } ( t )$ i<sub>s a non-</sub>d<sub>ecreas</sub>i<sub>ng</sub> f<sub>unc</sub>ti<sub>on an</sub>d $F _ { 1 } ^ { + } ( t ) \leq F _ { 2 } ^ { + } ( t )$ :

$$
\begin{array} { l } { { C _ { 1 , 2 } = \displaystyle \int _ { - \infty } ^ { \infty } F _ { 2 } ^ { - } ( t ) d F _ { 1 } ^ { + } ( t ) = \displaystyle [ F _ { 2 } ^ { - } ( t ) F _ { 1 } ^ { + } ( t ) ] _ { - \infty } ^ { \infty } - \displaystyle \int _ { - \infty } ^ { \infty } F _ { 1 } ^ { + } ( t ) d F _ { 2 } ^ { - } ( t ) } } \\ { { \mathrm { ~ } = 1 - \displaystyle \int _ { - \infty } ^ { \infty } F _ { 1 } ^ { + } ( t ) d F _ { 2 } ^ { - } ( t ) \ge 1 - \displaystyle \int _ { - \infty } ^ { \infty } F _ { 2 } ^ { + } ( t ) d F _ { 2 } ^ { - } ( t ) = C _ { 2 } } } \end{array}
$$

## 2. Upper Bound: $C _ { 1 , 2 } \leq C _ { 1 }$

B<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 1<sub>:</sub> $F _ { 2 } ^ { - } ( t ) \geq F _ { 1 } ^ { - } ( t )$ , ∀t (controls in $S _ { 1 }$ h<sub>ave</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>ca</sub>ll<sub>y</sub> l<sub>ower</sub> <sub>scores</sub> th<sub>an</sub> <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> i<sub>n</sub> $S _ { 2 } )$ :

$$
C _ { 1 , 2 } = \int _ { - \infty } ^ { \infty } F _ { 2 } ^ { - } ( t ) d F _ { 1 } ^ { + } ( t ) \le \int _ { - \infty } ^ { \infty } F _ { 1 } ^ { - } ( t ) d F _ { 1 } ^ { + } ( t ) = C _ { 1 }
$$

Th<sub>us,</sub> $C _ { 2 } \leq C _ { 1 , 2 } \leq C _ { 1 }$ <sub>.</sub> Si<sub>m</sub>l<sub>ar</sub>l<sub>y, we can s</sub>h<sub>ow</sub> $C _ { 2 } \leq C _ { 2 , 1 } \leq C _ { 1 }$

## 9.3 Main Theorem

Proposition 3 (Ordering of Localized AUC Expectations Across Model Competence Regions). If Assumptions 1 and 2 hold, then the population regional expectations of localized AUC functionals satisfy:

$$
\mathbb { E } _ { X _ { i } \sim P _ { X | S _ { 1 } } } [ \mathrm { L A U C } ( X _ { i } ) ] > \mathbb { E } _ { X _ { j } \sim P _ { X | S _ { 2 } } } [ \mathrm { L A U C } ( X _ { j } ) ]
$$

Proof.

(61)

$$
\begin{array} { r l } { \mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } \left[ \mathrm { L A U C } ( X _ { i } ) \right] = \mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } \left[ \frac { \mathbb { E } _ { X ^ { + } \sim P ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( X ^ { + } , X _ { i } ) w ( X ^ { - } , X _ { i } ) \mathbb { I } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) \right] } { W _ { i } } \right] } & { } \\ { = \mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } \left[ \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ \frac { w ( X ^ { + } , X _ { i } ) w ( X ^ { - } , X _ { i } ) \mathbb { I } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) } { W _ { i } } \right] \right] } & { } \\ { = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ \mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } \left[ \frac { w ( X ^ { + } , X _ { i } ) w ( X ^ { - } , X _ { i } ) \mathbb { I } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) } { W _ { i } } \right] \right] } & { } \\ { = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) \mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } \left[ \frac { w ( X ^ { + } , X _ { i } ) w ( X ^ { - } , X _ { i } ) } { W _ { i } } \right] \right] } & { } \end{array}\tag{62}
$$

(63)

(64)

Si<sub>m</sub>il<sub>ar</sub>l<sub>y,</sub>

$$
\mathbb { E } _ { X _ { j } \sim P _ { X _ { j } \in S _ { 1 } } } \left[ \mathrm { L A U C } ( X _ { j } ) \right] = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ 1 \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) \mathbb { E } _ { X _ { j } \sim P _ { X _ { j } \in S _ { 1 } } } \left[ \frac { w ( X ^ { + } , X _ { j } ) w ( X ^ { - } , X _ { j } ) } { W _ { j } } \right] \right] .\tag{65}
$$

L<sub>e</sub>t $\begin{array} { r } { D _ { 1 } ( X _ { i } ) = \mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } \left[ \frac { w ( X ^ { + } , X _ { i } ) w ( X ^ { - } , X _ { i } ) } { W _ { i } } \right] } \end{array}$ <sub>.</sub> N<sub>ow</sub> b<sub>y</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>on our expec</sub>t<sub>a</sub>ti<sub>ons w</sub>ith $S _ { 1 } \times S _ { 1 }$ <sub>an</sub>d $S _ { 2 } \times S _ { 2 }$ <sub>we</sub> fi<sub>n</sub>d<sub>:</sub>

$$
\begin{array} { r l } & { \mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } [ \mathrm { L A U C } ( X _ { i } ) ] = } \\ & { = \mathbb { E } _ { X + \sim F ^ { + } , X ^ { - } \sim F ^ { - } } [ \mathbb { 1 } ( f ( X ^ { + } ) > f ( X ^ { - } ) ) D _ { 1 } ( X _ { i } ) \middle | ( X ^ { + } , X ^ { - } ) \in S _ { 1 } \times S _ { 1 } ] \mathbb { P } ( ( X ^ { + } , X ^ { - } ) \in S _ { 1 } \times S _ { 1 } ) + } \\ & { + \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } [ \mathbb { 1 } ( f ( X ^ { + } ) > f ( X ^ { - } ) ) D _ { 1 } ( X _ { i } ) \middle | ( ( X ^ { + } , X ^ { - } ) \in ( S _ { 1 } \times S _ { 1 } ) ^ { C } \middle | \mathbb { P } ( ( X ^ { + } , X ^ { - } ) \in ( S _ { 1 } \times S _ { 1 } ) ^ { C } )   } \end{array}\tag{66}
$$

(67)

(68)

Si<sub>m</sub>il<sub>ar</sub>l<sub>y,</sub> if $\begin{array} { r } { D _ { 2 } ( X _ { j } ) = \mathbb { E } _ { X _ { j } \sim P _ { X _ { j } \in S _ { 2 } } } \left[ \frac { w ( X ^ { + } , X _ { j } ) w ( X ^ { - } , X _ { j } ) } { W _ { j } } \right] } \end{array}$ <sub>,</sub> th<sub>en we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } & { \mathbb { E } _ { X _ { j } \sim F _ { X _ { j } \in \mathcal { S } _ { 2 } } } \left[ { \mathrm { L A U C } } ( X _ { j } ) \right] = \qquad } & { \mathrm { ( } } \\ & { = \mathbb { E } _ { X + \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ { \mathbb { 1 } } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) D _ { 2 } ( X _ { j } ) \vert ( X ^ { + } , X ^ { - } ) \in S _ { 2 } \times S _ { 2 } \right] \mathbb { P } \left( ( X ^ { + } , X ^ { - } ) \in S _ { 2 } \times S _ { 2 } \right) + \qquad \mathrm { ( } } \\ & { + \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ { \mathbb { 1 } } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) D _ { 2 } ( X _ { j } ) \vert ( ( X ^ { + } , X ^ { - } ) \in ( S _ { 2 } \times S _ { 2 } ) ^ { C } \right) \mathbb { P } \left( ( X ^ { + } , X ^ { - } ) \in ( S _ { 2 } \times S _ { 2 } ) ^ { C } \right) } \end{array}\tag{69}
$$

70)

(71)

B<sub>y</sub> A<sub>ssump</sub>ti<sub>on</sub> 2 9<sub>.</sub>1 <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } & { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) D _ { 1 } ( X _ { i } ) | ( X ^ { + } , X ^ { - } ) \in S _ { 1 } \times S _ { 1 } \right] \mathbb { P } \left( ( X ^ { + } , X ^ { - } ) \in S _ { 1 } \times S _ { 1 } \right) \geq } \\ & { \geq \left( \epsilon - ( 1 - C _ { 1 } ) \right) D _ { 1 } \left( \frac { 1 } { 4 } - \eta \right) . } \end{array}\tag{72}
$$

(73)

$$
\begin{array} { r } { \mathbb { E } _ { X ^ { + } \sim { F ^ { + } } , X ^ { - } \sim F ^ { - } } [ \mathbb { I } ( f ( X ^ { + } ) > f ( X ^ { - } ) ) D _ { 1 } ( X _ { i } ) \big | \big ( ( X ^ { + } , X ^ { - } ) \in ( S _ { 1 } \times S _ { 1 } ) ^ { \mathcal { C } } \big ] \mathbb { P } ( ( X ^ { + } , X ^ { - } ) \in ( S _ { 1 } \times S _ { 1 } ) ^ { \mathcal { C } } ) \geq 0 . } \end{array}\tag{74}
$$

H<sub>ence, we</sub> h<sub>ave</sub>

$$
\mathbb { E } _ { X _ { i } \sim P _ { X _ { i } \in S _ { 1 } } } \left[ \mathrm { L A U C } ( X _ { i } ) \right] \geq ( \epsilon - ( 1 - C _ { 1 } ) ) D _ { 1 } \left( { \frac { 1 } { 4 } } - \eta \right) .\tag{75}
$$

Si<sub>m</sub>il<sub>ar</sub>l<sub>y,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r } { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) D _ { 2 } ( X _ { j } ) | ( X ^ { + } , X ^ { - } ) \in S _ { 2 } \times S _ { 2 } \right] \mathbb { P } \left( ( X ^ { + } , X ^ { - } ) \in S _ { 2 } \times S _ { 2 } \right) \le } \end{array}
$$

$$
\leq C _ { 2 } \left( { \frac { 1 } { 4 } } + \eta \right) .\tag{76}
$$

(77)

<sub>an</sub>d

$$
\begin{array} { r } { \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) D _ { 2 } ( X _ { j } ) | ( ( X ^ { + } , X ^ { - } ) \in ( S _ { 2 } \times S _ { 2 } ) ^ { C } \right] \mathbb { P } \left( ( X ^ { + } , X ^ { - } ) \in ( S _ { 2 } \times S _ { 2 } ) ^ { C } \right) \leq } \end{array}\tag{78}
$$

$$
\leq ( C _ { 2 } ^ { c r o s s } ( 1 - \epsilon ) + C _ { 2 } ^ { c r o s s } D _ { 2 } \epsilon ) \left( \frac { 1 } { 4 } + \eta \right) .\tag{79}
$$

H<sub>ence,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\mathbb { E } _ { X _ { j } \sim P _ { X _ { j } \in S _ { 2 } } } \left[ \mathrm { L A U C } ( X _ { j } ) \right] \leq C _ { 2 } \left( \frac { 1 } { 4 } + \eta \right) + ( C _ { 2 } ^ { c r o s s } ( 1 - \epsilon ) + C _ { 2 } ^ { c r o s s } D _ { 2 } \epsilon ) \left( \frac { 1 } { 4 } + \eta \right) .\tag{80}
$$

S<sub>o</sub> if $\begin{array} { r } { ( \epsilon - ( 1 - C _ { 1 } ) ) D _ { 1 } \left( \frac { 1 } { 4 } - \eta \right) \ge ( C _ { 2 } + ( C _ { 2 } ^ { c r o s s } ( 1 - \epsilon ) + C _ { 2 } ^ { c r o s s } D _ { 2 } \epsilon ) \left( \frac { 1 } { 4 } + \eta \right) } \end{array}$ <sub>,</sub> th<sub>en</sub> <sub>we</sub> <sub>are</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>e</sub>d<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>par</sub>t <sub>o</sub>f A<sub>ssump</sub>ti<sub>on</sub> 2<sub>.</sub>

Remark 1. Note thatfor example

$$
D _ { 1 } = 1 - \frac { \delta \gamma } { 4 ( 2 C _ { 1 } \delta + \gamma ) }\tag{81}
$$

$$
D _ { 2 } = \frac { \delta \gamma } { 2 ( \gamma - 1 ) ( 2 \gamma - \delta ) }\tag{82}
$$

$$
\epsilon = 1 - \frac { \delta } { 2 \gamma }\tag{83}
$$

$$
\delta \gamma
$$

$$
\eta = \frac { \textsf { - } 1 } { 4 ( \gamma ( 1 6 C _ { 1 } - 3 \delta ) - 8 \delta }\tag{84}
$$

$$
\delta = C _ { 1 } - C _ { 2 } , \gamma = 1 + C _ { 2 } ^ { c r o s s }\tag{85}
$$

would make

$$
( \epsilon - ( 1 - C _ { 1 } ) ) D _ { 1 } \left( \frac { 1 } { 4 } - \eta \right) \ge ( C _ { 2 } + ( C _ { 2 } ^ { c r o s s } ( 1 - \epsilon ) + C _ { 2 } ^ { c r o s s } D _ { 2 } \epsilon ) \left( \frac { 1 } { 4 } + \eta \right)\tag{86}
$$

true.

## 9.4 Estimator Construction and Asymptotic Consistency

L<sub>e</sub>t <sub>us</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> k<sub>erne</sub>l<sub>:</sub>

$$
\psi ( x , y ) = w ( x _ { i } , x ) w ( x _ { i } , y ) \mathbb { 1 } \left( f ( x ) > f ( y ) \right)\tag{87}
$$

L<sub>e</sub>t <sub>us</sub> <sub>a</sub>l<sub>so</sub> <sub>say,</sub>

$$
U _ { K , L } = \frac { 1 } { K L } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } w ( x _ { i } , X ^ { + } ) w ( x _ { i } , X ^ { - } ) \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right)\tag{88}
$$

$$
= { \frac { 1 } { K L } } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } \psi ( x _ { k } , x _ { l } ) \quad { \mathrm { a n d } } ,\tag{89}
$$

$$
W _ { + } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } w ( x _ { i } , x _ { k } ^ { + } ) \quad \mathrm { a n d } ,\tag{90}
$$

$$
W _ { - } = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } w ( x _ { i } , x _ { l } ^ { - } ) .\tag{91}
$$

L<sub>e</sub>t <sub>us assume</sub> th<sub>a</sub>t $N = K + L \to \infty { \mathrm { ~ a n d ~ } } { \frac { K } { N } } \to \lambda \in ( 0 , 1 )$ <sub>.</sub> Th<sub>en</sub> l<sub>e</sub>t

$$
\mu _ { U } = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( x _ { i } , X ^ { + } ) w ( x _ { i } , X ^ { - } ) \mathbb { 1 } \left( f ( X ^ { + } ) > f ( X ^ { - } ) \right) \right] = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ \psi ( X ^ { + } , X ^ { - } ) \right]\tag{92}
$$

$$
\mu _ { + } = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( x _ { i } , X ^ { + } ) \right]\tag{93}
$$

$$
\mu _ { - } = \mathbb { E } _ { X ^ { + } \sim F ^ { + } , X ^ { - } \sim F ^ { - } } \left[ w ( x _ { i } , X ^ { - } ) \right] .\tag{94}
$$

With thi<sub>s no</sub>t<sub>a</sub>ti<sub>on</sub> $\begin{array} { r } { \operatorname { L A U C } ( x _ { i } ) = \frac { \mu _ { U } } { \mu _ { + } \mu _ { - } } } \end{array}$ <sub>an</sub>d $\begin{array} { r } { \widehat { \mathrm { L A U C } } ( x _ { i } ) = \frac { U _ { K L } } { W _ { + } W _ { - } } } \end{array}$

## Regularity Assumptions:

• (A1) Finite<sub>,</sub> non-zero mean: $0 < \mu _ { + } , \mu _ { - } < \infty$

• (A2) Finite<sub>,</sub> non-zero variance: $0 < \operatorname { V a r } ( X ^ { + } ) , \operatorname { V a r } < \infty$

• (A3) Non-de<sub>g</sub>enerac<sub>y</sub>: $\begin{array} { r } { \operatorname { V a r } _ { X ^ { + } \sim F ^ { + } } ( \mathbb { E } _ { X ^ { - } \sim F ^ { - } } ( \psi ( X ^ { + } , X ^ { - } ) ) ) , \operatorname { V a r } _ { X ^ { - } \sim F ^ { - } } ( \mathbb { E } _ { X ^ { + } \sim F ^ { + } } ( \psi ( X ^ { + } , X ^ { - } ) ) ) > 0 . } \end{array}$

Proposition 4. If we follow Definitions 1 and 2, then the $\widehat { \mathrm { L A U C } }$ estimator is an unbiased consistent estimator of LAUC.

Proof. We will rely on two-sample U-statistics theory to derive asymptotic properties of $\widehat { \mathrm { L A U C } }$ <sub>.</sub> L<sub>e</sub>t <sub>us</sub> <sub>s</sub>t<sub>ar</sub>t b<sub>y</sub> l<sub>oo</sub>ki<sub>ng</sub> <sub>a</sub>t <sub>our</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>or:</sub>

$$
\begin{array} { r l r } & { } & { \widehat { \mathrm { L A U C } } ( x _ { i } ) = \frac { \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } w \left( x _ { i } , x _ { k } ^ { + } \right) w \left( x _ { i } , x _ { l } ^ { - } \right) \mathbb { 1 } \left( f ( x _ { k } ^ { + } ) > f ( x _ { l } ^ { - } ) \right) } { \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } w \left( x _ { i } , x _ { k } ^ { + } \right) w \left( x _ { i } , x _ { l } ^ { - } \right) } } \\ & { } & { = \frac { \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { L } w \left( x _ { i } , x _ { k } ^ { + } \right) w \left( x _ { i } , x _ { l } ^ { - } \right) \mathbb { 1 } \left( f ( x _ { k } ^ { + } ) > f ( x _ { l } ^ { - } ) \right) } { \left( \sum _ { k = 1 } ^ { K } w \left( x _ { i } , x _ { k } ^ { + } \right) \right) \left( \sum _ { l = 1 } ^ { L } w \left( x _ { i } , x _ { l } ^ { - } \right) \right) } } \end{array}\tag{95}
$$

(96)

B<sub>y</sub> the Weak Law of Lar<sub>g</sub>e Numbers (WLLN) for sam<sub>p</sub>le avera<sub>g</sub>es:

$$
W _ { + } \stackrel { p } {  } \mu _ { + } \quad \mathrm { a n d } ,\tag{97}
$$

$$
W _ { - } \stackrel { p } { \to } \mu _ { - } .\tag{98}
$$

B<sub>y</sub> the Law of Large Numbers for Two-Sample U-Statistics, $U _ { K , L } \stackrel { p } { \to } \mu _ { U }$

B<sub>y</sub> C<sub>o</sub>ntin<sub>uous</sub> M<sub>app</sub>in<sub>g</sub> Th<sub>eo</sub>r<sub>e</sub>m <sub>o</sub>n f<sub>u</sub>n<sub>c</sub>ti<sub>o</sub>n $\begin{array} { r } { g ( u , v , w ) = \frac { u } { v w } } \end{array}$ (continuous at $\begin{array} { r } { \mu _ { + } , \mu _ { - } > 0 ) , \frac { U _ { K , L } } { W _ { + } W _ { - } } \vec {  } \frac { \mu _ { U } } { \mu _ { + } \mu _ { - } } = } \end{array}$ LAUC.

Proposition 5 (Asymptotic Normality).

$$
\frac { 1 } { N } ( \widehat { \mathrm { L A U C } } ( x _ { i } ) - \mathrm { L A U C } ( x _ { i } ) ) \overset { d } {  } \mathcal { N } ( 0 , \frac { \sigma ^ { 2 } } { ( \mu _ { + } \mu _ { - } ) ^ { 2 } } ) , \quad w h e r e \quad \sigma ^ { 2 } = \frac { \sigma _ { + } ^ { 2 } } { \lambda } + \frac { \sigma _ { - } ^ { 2 } } { 1 - \lambda } .\tag{5}
$$

Proof. By asymptotic normality of generalized two sample U-statistic [18], we have

$$
\sqrt { N } ( U _ { K , L } - \mu _ { U } ) \stackrel { d } {  } \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \quad \mathrm { w h e r e } \quad \sigma ^ { 2 } = \frac { \sigma _ { + } ^ { 2 } } { \lambda } + \frac { \sigma _ { - } ^ { 2 } } { 1 - \lambda } .\tag{99}
$$

I<sub>n</sub> thi<sub>s</sub> <sub>equa</sub>ti<sub>on</sub> $\lambda = \operatorname* { l i m } _ { N \to \infty } K / N$ <sub>an</sub>d

$$
\begin{array} { r } { \sigma _ { + } ^ { 2 } = \operatorname { V a r } _ { X ^ { + } \sim F ^ { + } } \left( \mathbb { E } _ { X ^ { - } \sim F ^ { - } } \left[ \psi ( X ^ { + } , X ^ { - } ) \right] \right) } \end{array}\tag{100}
$$

$$
\sigma _ { - } ^ { 2 } = \operatorname { V a r } _ { X ^ { - } \sim F ^ { - } } \left( \operatorname { \mathbb { E } } _ { X ^ { + } \sim F ^ { + } } \left[ \psi ( X ^ { + } , X ^ { - } ) \right] \right) .\tag{101}
$$

B<sub>y</sub> Slutzk<sub>y</sub> [19], we have

$$
\sqrt { N } ( \widehat { \mathrm { L A U C } } ( x _ { i } ) - \mathrm { L A U C } ( x _ { i } ) ) = \sqrt { N } \left( \frac { U _ { K , L } } { W _ { + } W _ { - } } - \frac { \mu _ { U } } { \mu _ { + } \mu _ { - } } \right)\tag{102}
$$

$$
= \sqrt { N } \left( \frac { U _ { K , L } - \mu _ { U } } { W _ { + } W _ { - } } + \left( \frac { \mu _ { U } } { W ^ { + } W ^ { - } } - \frac { \mu _ { U } } { \mu _ { + } \mu _ { - } } \right) \right)\tag{103}
$$

$$
= \left( \frac { \sqrt { N } ( U _ { K , L } - \mu _ { U } ) } { W _ { + } W _ { - } } + \left( \frac { \mu _ { U } } { W _ { + } W _ { - } } - \frac { \mu _ { U } } { \mu _ { + } \mu _ { - } } \right) \right)\tag{104}
$$

$$
= \mathcal { N } ( 0 , \sigma ^ { 2 } / ( \mu _ { + } \mu _ { - } ) ^ { 2 } ) .\tag{105}
$$

Proposition 6. An asymptotically valid $1 0 0 ( 1 - \alpha ) \%$ confidence interval is as follows as $N \to \infty ;$

$$
\mathbb { P } ( \mathrm { L A U C } \in [ \widehat { \mathrm { L A U C } } _ { N } - z _ { 1 - \alpha / 2 } \frac { \widehat \sigma _ { \mathrm { L A U C } } } { \sqrt { N } } , \widehat { \mathrm { L A U C } } _ { N } + z _ { 1 - \alpha / 2 } \frac { \widehat \sigma _ { \mathrm { L A U C } } } { \sqrt { N } } ] )  1 - \alpha
$$

where $z _ { 1 - \alpha / 2 }$ is the standard normal critical value.

Proof. We aim to build an etimator $\sigma ^ { 2 } / ( \mu _ { + } \mu _ { - } ) ^ { 2 }$ <sub>, an</sub>d <sub>on</sub> thi<sub>s pa</sub>th <sub>we nee</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>ors o</sub>f $\sigma _ { + } / ( \mu _ { + } \mu _ { - } ) ^ { 2 }$ <sub>an</sub>d $\sigma _ { - } / ( \mu _ { + } \mu _ { - } ) ^ { 2 }$ <sub>.</sub> L<sub>e</sub>t’<sub>s ca</sub>ll th<sub>e es</sub>ti<sub>ma</sub>t<sub>or o</sub>f $\sigma ^ { 2 } / ( \mu _ { + } \mu _ { - } ) ^ { 2 }$ as $\hat { \sigma } _ { \mathrm { L A U C } }$ <sub>an</sub>d th<sub>e es</sub>ti<sub>ma</sub>t<sub>or o</sub>f $\sigma _ { + } / ( \mu _ { + } \mu _ { - } ) ^ { 2 }$ <sub>an</sub>d $\sigma _ { - } / ( \mu _ { + } \mu _ { - } ) ^ { 2 }$ $\hat { \sigma } _ { + , \mathrm { L A U C } } ^ { 2 }$ <sub>an</sub>d $\hat { \sigma } _ { - , \mathrm { L A U C } } ^ { 2 }$ <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> A<sub>n es</sub>ti<sub>ma</sub>t<sub>or o</sub>f $\mathbb { E } _ { X ^ { - } \sim F ^ { - } } \left[ \psi ( X ^ { + } , X ^ { - } ) \right]$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\psi ^ { + } ( x _ { k } ^ { + } ) = w ( x _ { i } , x _ { k } ^ { + } ) \left( \frac { 1 } { L } \sum _ { l = 1 } ^ { L } w ( x _ { i } , x _ { l } ^ { - } ) \mathbb { 1 } \left( f ( x _ { k } ^ { + } ) > f ( x _ { l } ^ { - } ) \right) \right) ,\tag{106}
$$

<sub>an</sub>d <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>l<sub>y an es</sub>ti<sub>ma</sub>t<sub>or o</sub>f $\mathbb { E } _ { X ^ { + } \sim F ^ { + } } \left[ \psi ( X ^ { + } , X ^ { - } ) \right]$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\psi ^ { - } ( x _ { l } ^ { - } ) = w ( x _ { i } , x _ { l } ^ { - } ) \left( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } w ( x _ { i } , x _ { k } ^ { + } ) \mathbb { 1 } \left( f ( x _ { k } ^ { + } ) > f ( x _ { l } ^ { - } ) \right) \right) .\tag{107}
$$

L<sub>e</sub>t <sub>us a</sub>l<sub>so</sub> d<sub>e</sub>fi<sub>ne:</sub>

$$
\hat { \mu } _ { + } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } w ( x _ { i } , x _ { k } ^ { + } ) \quad \mathrm { a n d } \quad \hat { \mu } _ { - } = \frac { 1 } { L } \sum _ { l = 1 } ^ { K } w ( x _ { i } , x _ { l } ^ { - } ) .\tag{108}
$$

H<sub>ence cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>ors o</sub>f th<sub>e var</sub>i<sub>ances are</sub>

$$
\widehat { \sigma } _ { \mathrm { + , L A U C } } ^ { 2 } = \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K } \left( \frac { \psi ^ { + } \left( x _ { k } ^ { + } \right) } { \widehat { \mu } _ { + } \widehat { \mu } _ { - } } - \widehat { \mathrm { L A U C } } _ { N } \right) ^ { 2 } \quad \mathrm { a n d } \quad \widehat { \sigma } _ { \mathrm { - , L A U C } } ^ { 2 } = \frac { 1 } { L - 1 } \sum _ { i = 1 } ^ { L } \left( \frac { \psi ^ { - } \left( x _ { i } ^ { - } \right) } { \widehat { \mu } _ { + } \widehat { \mu } _ { - } } - \widehat { \mathrm { L A U C } } _ { N } \right) ^ { 2 }
$$

$$
\hat { \sigma } _ { \mathrm { L A U C } } ^ { 2 } = \frac { N } { K } \hat { \sigma } _ { + , \mathrm { L A U C } } + \frac { N } { L } \hat { \sigma } _ { - , \mathrm { L A U C } }\tag{109}
$$

(110)

Fi<sub>na</sub>ll<sub>y</sub> th<sub>e</sub> <sub>asymp</sub>t<sub>o</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>va</sub>lid $1 0 0 ( 1 - \alpha ) \%$ <sub>con</sub>fid<sub>ence</sub> i<sub>n</sub>t<sub>erva</sub>l i<sub>s as</sub> f<sub>o</sub>ll<sub>ows as</sub> $N  \infty \mathrm { : }$

$$
\mathbb { P } ( \mathrm { L A U C } \in [ \widehat { \mathrm { L A U C } } _ { N } - z _ { 1 - \alpha / 2 } \frac { \widehat \sigma _ { \mathrm { L A U C } } } { \sqrt { N } } , \widehat { \mathrm { L A U C } } _ { N } + z _ { 1 - \alpha / 2 } \frac { \widehat \sigma _ { \mathrm { L A U C } } } { \sqrt { N } } ] )  1 - \alpha
$$

<sub>w</sub>h<sub>ere</sub> $z _ { 1 - \alpha / 2 }$ i<sub>s</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>norma</sub>l <sub>cr</sub>iti<sub>ca</sub>l <sub>va</sub>l<sub>ue.</sub>

## 9.5 Test of homogeneous predicted risk

T<sub>o</sub> t<sub>es</sub>t <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>r</sub>i<sub>s</sub>k i<sub>s</sub> fl<sub>a</sub>t i<sub>n</sub> th<sub>e</sub> l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group</sub> $S$ <sub>compare</sub>d t<sub>o</sub> it<sub>s</sub> <sub>comp</sub>l<sub>emen</sub>t $S ^ { c }$ , we f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> t<sub>es</sub>t <sub>as:</sub>

$$
H _ { 0 } : \frac { \sigma _ { S } ^ { 2 } } { \sigma _ { S ^ { c } } ^ { 2 } } \geq \delta \quad \mathrm { v s . } \quad H _ { A } : \frac { \sigma _ { S } ^ { 2 } } { \sigma _ { S ^ { c } } ^ { 2 } } < \delta\tag{111}
$$

<sub>w</sub>h<sub>ere</sub> $S$ i<sub>s</sub> th<sub>e</sub> l<sub>ow-per</sub>f<sub>ormance</sub> <sub>su</sub>b<sub>group,</sub> $S ^ { c }$ i<sub>s</sub> it<sub>s</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>,</sub> <sub>an</sub>d $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ <sub>represen</sub>t<sub>s</sub> th<sub>e non-</sub>fl<sub>a</sub>t<sub>ness</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>ra</sub>ti<sub>o.</sub>

Si<sub>nce</sub> $S$ <sub>an</sub>d $S ^ { c }$ do not share common observations, we can use a one-sided F-test to test $H _ { 0 }$ <sub>.</sub> L<sub>e</sub>t $n _ { S }$ <sub>an</sub>d $n _ { S ^ { c } }$ b<sub>e</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>s</sub>i<sub>zes</sub> <sub>o</sub>f th<sub>e</sub> <sub>respec</sub>ti<sub>ve</sub> <sub>reg</sub>i<sub>ons,</sub> <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>var</sub>i<sub>ances</sub> <sub>o</sub>f l<sub>og</sub>it<sub>-</sub>t<sub>rans</sub>f<sub>orme</sub>d <sub>pre</sub>di<sub>c</sub>t<sub>e</sub>d <sub>r</sub>i<sub>s</sub>k<sub>s</sub> <sub>as:</sub>

$$
s _ { S } ^ { 2 } = \mathrm { V a r } \left( \mathrm { l o g i t } ( \hat { p } _ { i } ) : i \in S \right) , \quad s _ { S ^ { c } } ^ { 2 } = \mathrm { V a r } \left( \mathrm { l o g i t } ( \hat { p } _ { i } ) : i \in S ^ { c } \right)\tag{112}
$$

D<sub>e</sub>fi<sub>ne</sub> th<sub>e o</sub>b<sub>serve</sub>d <sub>var</sub>i<sub>ance ra</sub>ti<sub>o:</sub>

$$
{ \hat { R } } = { \frac { s _ { S } ^ { 2 } } { s _ { S ^ { c } } ^ { 2 } } }\tag{113}
$$

At th<sub>e</sub> b<sub>oun</sub>d<sub>ary</sub> <sub>o</sub>f th<sub>e</sub> <sub>nu</sub>ll h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> $\begin{array} { r } { ( R = \frac { \sigma _ { S } ^ { 2 } } { \sigma _ { S ^ { c } } ^ { 2 } } = \delta ) } \end{array}$ <sub>,</sub> th<sub>e</sub> t<sub>es</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>c</sub> <sub>sca</sub>l<sub>e</sub>d b<sub>y</sub> $\delta$ follows an F-distribution:

$$
\frac { s _ { S } ^ { 2 } } { \delta s _ { S ^ { c } } ^ { 2 } } \sim F _ { n _ { S } - 1 , n _ { S ^ { c } } - 1 }\tag{114}
$$

Because the alternative h<sub>y</sub>pothesis posits a smaller variance in S, this is a lower-tail test. The p-value is given b<sub>y</sub>:

$$
p = P \left( F _ { n _ { S } - 1 , n _ { S } c - 1 } \leq { \frac { s _ { S } ^ { 2 } } { \delta s _ { S } ^ { 2 } c } } \right)\tag{115}
$$

We reject $H _ { 0 }$ at significance level α when:

$$
{ \frac { s _ { S } ^ { 2 } } { \delta s _ { S ^ { c } } ^ { 2 } } } \leq F _ { \alpha ; n _ { S } - 1 , n _ { S ^ { c } } - 1 }\tag{116}
$$

E<sub>qu</sub>i<sub>va</sub>l<sub>en</sub>tl<sub>y,</sub> <sub>we</sub> <sub>can</sub> <sub>cons</sub>t<sub>ruc</sub>t <sub>a</sub> $1 0 0 ( 1 - \alpha ) \%$ <sub>one-s</sub>id<sub>e</sub>d <sub>upper con</sub>fid<sub>ence</sub> b<sub>oun</sub>d $R _ { U }$ f<sub>or</sub> th<sub>e</sub> t<sub>rue var</sub>i<sub>ance ra</sub>ti<sub>o</sub> $\begin{array} { r } { R = \frac { \sigma _ { S } ^ { 2 } } { \sigma _ { S ^ { c } } ^ { 2 } } } \end{array}$ :

$$
R _ { U } = \frac { \hat { R } } { F _ { \alpha ; n _ { S } - 1 , n _ { S } c - 1 } }\tag{117}
$$

and reject $H _ { 0 }$ if $R _ { U } < \delta$

## Assumptions

leftmirgin=\* Normality within each region: log $\mathsf { i t } ( p ) \mid S \sim \mathcal { N } ( \mu _ { S } , \sigma _ { S } ^ { 2 } )$ and logit(p) | $S ^ { c } \sim \mathcal N ( \mu _ { S ^ { c } } , \sigma _ { S ^ { c } } ^ { 2 } )$

leftmiirgiin=\* Independence of observations: Predictions must be independent across subjects. This holds <sub>un</sub>d<sub>er</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>.</sub>i<sub>.</sub>d<sub>.</sub> <sub>samp</sub>li<sub>ng,</sub> b<sub>u</sub>t <sub>c</sub>l<sub>us</sub>t<sub>ere</sub>d <sub>or</sub> l<sub>ong</sub>it<sub>u</sub>di<sub>na</sub>l <sub>se</sub>tti<sub>ngs</sub> <sub>requ</sub>i<sub>re</sub> bl<sub>oc</sub>k <sub>resamp</sub>li<sub>ng</sub> adjustments.

leftmiiirgiiin=\* Independence of the two variance estimates: Because S and $S ^ { c }$ <sup>are non-o</sup>v<sup>erla</sup>pp<sup>in</sup>g <sup>sets</sup> <sub>o</sub>f i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> $s _ { S } ^ { 2 }$ <sub>an</sub>d $s _ { S ^ { c } } ^ { 2 }$ are statisticall<sub>y</sub> independent. Comparing S with its com<sub>p</sub><sup>l</sup>ement $S ^ { c }$ is therefore cleaner than comparing S with the entire dataset.

leftmivrgivn=\* Hold-out validation required for S: The region S must be fixed independently of the data used for the test. If an algorithm searches the same sample to discover S and evaluate the F-test, selection bias invalidates the nominal F-distribution. This test must be conducted on an i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t h<sub>o</sub>ld<sub>-ou</sub>t d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub>

## 9.6 Supplementary Figures and Tables

Figure 9 shows the attrition statistics of the NHANES data 1999-2010. We initiall<sub>y</sub> identified 62, 160 respondents from the NHANES 1999–2010 c<sub>y</sub>cles (A–F). We restricted the sam<sub>p</sub>le to adults a<sub>g</sub>ed 30–79 <sub>y</sub>ears (the a<sub>g</sub>e ran<sub>g</sub>e for the PREVENT equations), leaving 23, 920 individuals. After excluding pregnant women $( n ~ = ~ 4 6 6 )$ <sub>an</sub>d individuals with self-reported prior cardiovascular disease (n = 2, 272), 21, 182 remained. We further excluded participants with missing data on an<sub>y</sub> of the 11 PREVENT predictor variables (n=3, 151), resulting in 18, 031 individuals with complete data. Of these, 259 were excluded because the preventr scoring algorithm flagged out-of-range input values, leaving 17, 772 valid risk scores. Finall<sub>y</sub>, we restricted to participants eligible for mortalit<sub>y</sub> linkage, <sub>y</sub>ielding the final anal<sub>y</sub>tical cohort of 17, 748 individuals.

![](images/834fe5b651e354a460d58bac472b419bbf7020ea1d91b10a8100d28741df5900.jpg)  
Figure 9. Attrition of NHANES data. Initially there are 62, 160 respondents from the NHANES 1999–2010 cycles (A–F). We restricted the sample to adults aged 30–79 <sub>y</sub>ears (the age range for the PREVENT equations), leaving 23, 920 individuals. After excluding pregnant women (n = 466) and individuals with self-reported prior cardiovascular disease (n = 2, 272), 21, 182 remained. We further excluded participants with missing data on an<sub>y</sub> of the 11 PREVENT predictor variables (n = 3, 151), resulting in 18, 031 individuals with complete data. Of these, 259 were excluded because the preventr scoring algorithm flagged out-of-range input values, leaving 17, 772 valid risk scores. Finall<sub>y</sub>, we restricted to participants eligible for mortalit<sub>y</sub> linkage, <sub>y</sub>ielding the final anal<sub>y</sub>tical cohort of 17, 748 individuals.

A <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>c o</sub>f th<sub>e</sub> NHANES d<sub>a</sub>t<sub>a</sub> b<sub>e</sub>f<sub>ore prepocess</sub>i<sub>ng</sub> i<sub>s s</sub>h<sub>own</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 2 <sub>an</sub>d 3<sub>.</sub>

UMAP <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> did <sub>no</sub>t <sub>revea</sub>l <sub>o</sub>b<sub>v</sub>i<sub>ous</sub> <sub>c</sub>l<sub>us</sub>t<sub>ers</sub> i<sub>n</sub> th<sub>e</sub> d<sub>a</sub>t<sub>a.</sub> B<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f <sub>our</sub> <sub>ma</sub>i<sub>n</sub> <sub>me</sub>th<sub>o</sub>d<sub>,</sub> <sub>we</sub> utilized a dimensionalit<sub>y</sub> <sub>p</sub>rojection method UMAP[20] to create more intuitive fi<sub>g</sub>ures of the in<sub>p</sub>ut data with res<sub>p</sub>ect to several baseline variables (a<sub>g</sub>e, sex, etc.).

After preprocessing (filter out participants with more than 40% variables missing and variables with more than 90% missing data), the distribution of several baseline variables were presented via UMAP visualization method. Fi<sub>g</sub>ure 10 shows the distribution of a<sub>g</sub>e, sex (male=1, female=0), education (Less Than 9th Grade=1, 9-11th Grade (Includes 12th <sub>g</sub>rade with no di<sub>p</sub>loma)=2, Hi<sub>g</sub>h School Grad/GED or Equivalent=3, Some Colle<sub>g</sub>e or AA de<sub>g</sub>ree=4, Colle<sub>g</sub>e Graduate or above=5, Refused=7, Don’t Know=9), race/ethnicit<sub>y</sub>(Mexican American=1, Other His<sub>p</sub>anic=2, Non-His<sub>p</sub>anic White=3, Non-His<sub>p</sub>anic Black=4, Other Race - Includin<sub>g</sub> Multi-Racial=5) and the <sub>ra</sub>ti<sub>o o</sub>f <sub>pover</sub>t<sub>y</sub> t<sub>o</sub> i<sub>ncome</sub> i<sub>n</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a se</sub>t<sub>.</sub> Fi<sub>gure</sub> 17 <sub>s</sub>h<sub>ows</sub> th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f <sub>same se</sub>t <sub>o</sub>f <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> i<sub>n</sub> t<sub>es</sub>ti<sub>ng</sub> d<sub>a</sub>t<sub>a</sub> <sub>se</sub>t<sub>.</sub>

![](images/9e4ab6475efbfc63b52325af3a8eb8f615586e1ce94bcccaad2e708ccb1424fe.jpg)  
Figure 10. Absence of UMAP clusters in NHANES datset. UMAP visualization of the distribution of age, sex (male=1, female=0), education (Less Than 9th Grade=1, 9-11th Grade (Includes 12th <sub>g</sub>rade with no di<sub>p</sub>loma)=2, Hi<sub>g</sub>h School Grad/GED or Equivalent=3, Some Colle<sub>g</sub>e or AA de<sub>g</sub>ree=4, Colle<sub>g</sub>e Graduate or above=5, Refused=7, Don’t Know=9), race/ethnicit<sub>y</sub>(Mexican American=1, Other His<sub>p</sub>anic=2, Non-His<sub>p</sub>anic White=3, Non-His<sub>p</sub>anic Black=4, Other Race - Includin<sub>g</sub> Multi-Racial=5) and the ratio of <sub>p</sub>overt<sub>y</sub> to income i<sub>n</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a se</sub>t<sub>.</sub> F<sub>rom</sub> th<sub>e</sub> fi<sub>gure</sub> it <sub>can</sub> b<sub>e seen</sub> th<sub>a</sub>t th<sub>ere</sub> i<sub>sn</sub>’t <sub>o</sub>b<sub>v</sub>i<sub>ous c</sub>l<sub>us</sub>t<sub>er o</sub>f <sub>su</sub>b<sub>groups.</sub>

## 9.7 Inclusion and Exclusion Criteria into External Validation Cohort.

Partici<sub>p</sub>ants a<sub>g</sub>ed 30 to 79 <sub>y</sub>ears at the time of baseline assessment (instance 0) in the UK Biobank were included, <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith PREVENT <sub>cr</sub>it<sub>er</sub>i<sub>a.</sub> P<sub>ar</sub>ti<sub>c</sub>i<sub>pan</sub>t<sub>s</sub> <sub>were</sub> <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d if <sub>pregnan</sub>t <sub>or</sub> if <sub>a</sub>t b<sub>ase</sub>li<sub>ne</sub> h<sub>ave</sub> <sub>repor</sub>t<sub>e</sub>d <sub>a</sub> hi<sub>s</sub>t<sub>ory</sub> of cardiovascular disease (CVD), includin<sub>g</sub> coronar<sub>y</sub> heart disease, m<sub>y</sub>ocardial infarction, or stroke. Partici<sub>p</sub>ants <sub>w</sub>ith <sub>m</sub>i<sub>ss</sub>i<sub>ng</sub> XGB<sub>oos</sub>t i<sub>npu</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>es were exc</sub>l<sub>u</sub>d<sub>e</sub>d t<sub>oo.</sub> All <sub>necessary c</sub>li<sub>n</sub>i<sub>ca</sub>l<sub>,</sub> l<sub>a</sub>b<sub>ora</sub>t<sub>ory, an</sub>d d<sub>emograp</sub>hi<sub>c</sub> <sub>pre</sub>di<sub>c</sub>t<sub>ors</sub> i<sub>n</sub> th<sub>e</sub> UK Bi<sub>o</sub>b<sub>an</sub>k <sub>were</sub> h<sub>armon</sub>i<sub>ze</sub>d t<sub>o</sub> <sub>a</sub>li<sub>gn</sub> <sub>w</sub>ith th<sub>e</sub> <sub>co</sub>di<sub>ng</sub> <sub>conven</sub>ti<sub>ons,</sub> <sub>measuremen</sub>t <sub>un</sub>it<sub>s,</sub> <sub>an</sub>d t<sub>runca</sub>ti<sub>on</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s</sub> <sub>use</sub>d i<sub>n</sub> th<sub>e</sub> NHANES t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>.</sub> H<sub>armon</sub>i<sub>za</sub>ti<sub>on</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> b<sub>e</sub>t<sub>ween</sub> NHANES <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> <sub>an</sub>d UK Bi<sub>o</sub>b<sub>an</sub>k <sub>are</sub> <sub>summar</sub>i<sub>ze</sub>d i<sub>n</sub> Fi<sub>gure</sub> 12<sub>.</sub>

## 9.8 Cardiovascular Endpoint Definition

W<sub>e</sub> <sub>va</sub>lid<sub>a</sub>t<sub>e</sub>d <sub>our</sub> <sub>mo</sub>d<sub>e</sub>l <sub>us</sub>i<sub>ng</sub> CVD d<sub>ea</sub>th <sub>s</sub>i<sub>nce</sub> thi<sub>s</sub> i<sub>s</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come</sub> <sub>w</sub>hi<sub>c</sub>h <sub>was</sub> <sub>use</sub>d i<sub>n</sub> th<sub>e</sub> NHANES d<sub>a</sub>t<sub>ase</sub>t t<sub>o</sub> tr<sub>a</sub>in LOCATOR m<sub>o</sub>d<sub>e</sub>l<sub>.</sub>

![](images/4bdc13a393049d3aa063ac57e8a1387434252341779431f446b590054af1af3b.jpg)  
Figure 11. Three ROC curves of three models split using 33% and 67% percentiles of XGBoost predictions logit $\left( \mathrm { L A U C } ( x _ { i } ) \right)$ <sup>S</sup>u<sup>b</sup><sub>g</sub>rou<sub>p</sub> $0 \% - 3 3 \%$ corres<sub>p</sub>on<sup>d</sup>s to <sup>l</sup>ow-<sub>p</sub>er<sup>f</sup>ormance su<sup>b</sup><sub>g</sub>rou<sub>p</sub>, <sup>S</sup>u<sup>b</sup><sub>g</sub>rou<sub>p</sub> $3 3 \% - 6 7 \%$ <sub>correspon</sub>d<sub>s</sub> t<sub>o me</sub>di<sub>um-per</sub>f<sub>ormance su</sub>b<sub>group,</sub> <sub>a</sub>nd S<sub>u</sub>b<sub>g</sub>r<sub>oup</sub> $6 7 \% - 1 0 0 \%$ <sub>correspon</sub>d<sub>s</sub> t<sub>o</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>group.</sub> Th<sub>e</sub> ROC <sub>curves o</sub>f th<sub>e</sub> th<sub>ree su</sub>b<sub>groups s</sub>h<sub>ows</sub> i<sub>ncreas</sub>i<sub>ng</sub> <sub>per</sub>f<sub>ormance w</sub>h<sub>en c</sub>li<sub>m</sub>bi<sub>ng</sub> f<sub>rom</sub> l<sub>ow-</sub> t<sub>o</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>group.</sub> Th<sub>e</sub> AUC <sub>va</sub>l<sub>ues s</sub>h<sub>own</sub> i<sub>n</sub> th<sub>e</sub> l<sub>egen</sub>d <sub>a</sub>l<sub>so va</sub>lid<sub>a</sub>t<sub>e</sub> thi<sub>s.</sub> With confidence level of 95%, the confidence band is shown in the figure and the confidence intervals of AUC values are shown in the legend.

<table><tr><td rowspan=1 colspan=1>NHANESPredictor</td><td rowspan=1 colspan=1>UKBiobankField</td><td rowspan=1 colspan=1>Mapping / Conversion Rule</td><td rowspan=1 colspan=1>Truncation /Constraints</td></tr><tr><td rowspan=1 colspan=1>Age</td><td rowspan=1 colspan=1>21003</td><td rowspan=1 colspan=1>Continuous (years)</td><td rowspan=1 colspan=1>Restricted to30-79 years</td></tr><tr><td rowspan=1 colspan=1>Sex</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>1 = Male, 0 = Female</td><td rowspan=1 colspan=1>Binary</td></tr><tr><td rowspan=1 colspan=1>Race / Ethnicity</td><td rowspan=1 colspan=1>21000</td><td rowspan=1 colspan=1>White, British, Irish, or any other White background → 3 (Non-Hispanic White);Black or Black British, Caribbean, African, or any other Blackbackground → 4 (Non-Hispanic Black);Asian or Asian British (Indian, Pakistani, Bangladeshi, Chinese),Mixed background, and other ethnic groups. → 5Asian/Mixed/Other</td><td rowspan=1 colspan=1>Categorical(1-5)</td></tr><tr><td rowspan=1 colspan=1>Poverty IncomeRatio</td><td rowspan=1 colspan=1>189(TDI)</td><td rowspan=1 colspan=1>Piecewise linear mapping from UK Biobank Townsend DeprivationIndex (TDI)PIR = $\textstyle \int { \frac { 1 1 . 0 - { \mathsf { T D } } { \mathsf { I } } } { 8 . 0 } } ,$     $\mathsf { i f } \mathsf { T D I } \geq 3 . 0 \left( \mathsf { m a p s } \mathsf { T D I } \in [ 3 . 0 , 1 1 . 0 ] \to \mathsf { P I R } \in [ 0 , 1 ] \right)$ 5.315-TDI  if TDI &lt; 3.0 (maps TDI ∈ [−, 6.263.0) → PIR ∈ (1, 5])2.315</td><td rowspan=1 colspan=1>Bounded to[0,5]</td></tr><tr><td rowspan=1 colspan=1>Education</td><td rowspan=1 colspan=1>Fields6138,845</td><td rowspan=1 colspan=1>Category 5 (College graduate or above): College or universitydegree.Category 4 (Some college / Associate degree): A/AS levels,NVQ/HND/HNC, or other professional qualifications (e.g.,nursing, teaching).Category 3 (High school / GED equivalent): O-levels/GCSEs orCSEs.Category 2 (9-11th grade): No formal qualification (6138 = -7)with age leaving full-time education ≥ 14years.Category 1 (&lt;9th grade): No formal qualification (6138 = -7) withage leaving full-time education &lt; 14years.</td><td rowspan=1 colspan=1>Categorical(1-5)</td></tr><tr><td rowspan=1 colspan=1>Systolic BP</td><td rowspan=1 colspan=1>4080,93</td><td rowspan=1 colspan=1>Average of baseline repeated readings (mmHg)</td><td rowspan=1 colspan=1>Capped at[90,200] mmH</td></tr><tr><td rowspan=1 colspan=1>Diastolic B</td><td rowspan=1 colspan=1>4079, 94 Av</td><td rowspan=1 colspan=1>erage of baseline readings (mmHg)</td><td rowspan=1 colspan=1>Continuous</td></tr><tr><td rowspan=1 colspan=1>AntihypertensiveTreatment</td><td rowspan=1 colspan=1>6153,20003</td><td rowspan=1 colspan=1>Prescribed blood pressure lowering therapy</td><td rowspan=1 colspan=1>1 = Yes, 0= No</td></tr><tr><td rowspan=1 colspan=1>Total Cholesterol</td><td rowspan=1 colspan=1>30690</td><td rowspan=1 colspan=1>mmol/L × 38.67 → mg/dL</td><td rowspan=1 colspan=1>Capped at[130,320] mg</td></tr><tr><td rowspan=1 colspan=1>HDL Cholesterol</td><td rowspan=1 colspan=1>30760</td><td rowspan=1 colspan=1>mmol/L × 38.67 → mg/dL</td><td rowspan=1 colspan=1>Capped at[20,100] mg/d</td></tr><tr><td rowspan=1 colspan=1>Statin Therapy</td><td rowspan=1 colspan=1>6153,20003</td><td rowspan=1 colspan=1>Prescribed cholesterol lowering therapy</td><td rowspan=1 colspan=1>1 = Yes, 0= No</td></tr><tr><td rowspan=1 colspan=1>BMI</td><td rowspan=1 colspan=1>21001</td><td rowspan=1 colspan=1>Continuous (kg/m²)</td><td rowspan=1 colspan=1>Capped at[18.5,39.9] kg</td></tr><tr><td rowspan=1 colspan=1>Diabetes</td><td rowspan=1 colspan=1>2443,20002</td><td rowspan=1 colspan=1>Self-reported or clinician-diagnosed diabetes</td><td rowspan=1 colspan=1>1 = Yes, 0=No</td></tr><tr><td rowspan=1 colspan=1>Current Smoking</td><td rowspan=1 colspan=1>Fields20116,20160</td><td rowspan=1 colspan=1>≥ 100 lifetime cigarettes and current smoker</td><td rowspan=1 colspan=1>1 = Yes, 0= No</td></tr><tr><td rowspan=1 colspan=1>Cancer History</td><td rowspan=1 colspan=1>2453,20001</td><td rowspan=1 colspan=1>History of any malignant neoplasm</td><td rowspan=1 colspan=1>1 = Yes, 0No</td></tr><tr><td rowspan=1 colspan=1>Hemoglobin</td><td rowspan=1 colspan=1>30020</td><td rowspan=1 colspan=1>Continuous (g/dL)</td><td rowspan=1 colspan=1>Continuous</td></tr><tr><td rowspan=1 colspan=1>Hematocrit</td><td rowspan=1 colspan=1>30030</td><td rowspan=1 colspan=1>Continuous (%)</td><td rowspan=1 colspan=1>Continuous</td></tr></table>

![](images/7fefc799d373553356797fa05f394d1a3774e85a3b8491129c0a543da7bd413d.jpg)

![](images/ce1bc23bc411d7c0a1418d36ec0b2b265df25b33273fceae0be35d791d5d2aec.jpg)

Figure 13. The histogram of standard AUC values of low- and high-performance subgroups in the test set of NHANES data with 100 seeds (0-99). The whole method is im<sub>p</sub>lemented: train the XGBoost model in the trainin<sub>g</sub> set, then identif<sub>y</sub> low- and hi<sub>g</sub>h-<sub>p</sub>erformance <sub>su</sub>b<sub>groups</sub> i<sub>n</sub> th<sub>e</sub> t<sub>es</sub>t <sub>se</sub>t<sub>.</sub> Th<sub>en ca</sub>l<sub>cu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e unwe</sub>i<sub>g</sub>ht<sub>e</sub>d AUC i<sub>n</sub> l<sub>ow- an</sub>d hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>groups respec</sub>ti<sub>ve</sub>l<sub>y.</sub> R<sub>epea</sub>t th<sub>e</sub> <sub>proce</sub>d<sub>ure</sub> f<sub>or</sub> 100 <sub>see</sub>d<sub>s</sub> f<sub>rom</sub> 0 t<sub>o</sub> 99 <sub>an</sub>d <sub>p</sub>l<sub>o</sub>t th<sub>e</sub> hi<sub>s</sub>t<sub>ogram us</sub>i<sub>ng a</sub>ll AUC <sub>va</sub>l<sub>ues.</sub> Th<sub>e</sub> fi<sub>gure s</sub>h<sub>ows</sub> th<sub>a</sub>t l<sub>ow-per</sub>f<sub>ormance su</sub>b<sub>group</sub> h<sub>as</sub> l<sub>ower</sub> AUC th<sub>an</sub> hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>group.</sub> Th<sub>e mean</sub> AUC <sub>o</sub>f l<sub>ow-per</sub>f<sub>ormance su</sub>b<sub>group</sub> i<sub>s</sub> 0<sub>.</sub>744<sub>, w</sub>ith <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> 0<sub>.</sub>0157<sub>.</sub> Th<sub>e mean</sub> AUC <sub>o</sub>f hi<sub>g</sub>h<sub>-per</sub>f<sub>ormance su</sub>b<sub>group</sub> i<sub>s</sub> 0<sub>.</sub>822<sub>, w</sub>ith <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>va</sub>ti<sub>a</sub>ti<sub>on</sub> 0<sub>.</sub>0394<sub>.</sub>  
![](images/864ef077eb14e0b9eef44853f6ab5d5b39797e57798cfe5704967c13c6f3e6a3.jpg)

Figure 14. UMAP visualization of the distribution of age, sex (male=1, female=0), education (Less Than 9th Grade=1, 9-11th Grade (Includes 12th <sub>g</sub>rade with no di<sub>p</sub>loma)=2, Hi<sub>g</sub>h School Grad/GED or Equivalent=3, Some Colle<sub>g</sub>e or AA de<sub>g</sub>ree=4, Colle<sub>g</sub>e Graduate or above=5, Refused=7, Don’t Know=9), race/ethnicit<sub>y</sub>(Mexican American=1, Other His<sub>p</sub>anic=2, Non-His<sub>p</sub>anic White=3, Non-Hispanic Black=4, Other Race - Including Multi-Racial=5) and the ratio of poverty to income in testing data set.  
![](images/5ac32296e2b7b980e261fc83a9d346416dc2520af01a79e26e06a9c17253afa8.jpg)  
Figure 15. The histogram of the calculated weighted AUC in test set. The chosen seed is 64, corresponding to the maximum AUC<sub>-</sub>d<sub>e</sub>lt<sub>a pars</sub>i<sub>mon</sub>i<sub>ous mo</sub>d<sub>e</sub>l<sub>.</sub> M<sub>ean</sub> AUC<sub>:</sub> 0<sub>.</sub>8260<sub>,</sub> M<sub>e</sub>di<sub>an</sub> AUC<sub>:</sub> 0<sub>.</sub>8282

![](images/21af8ec1a8979e5e62093ca431692d8a3d1e5f301b98ce2ebadd63eb340efec2.jpg)  
Figure 16. The histogram of the calculated weighted AUC in train set. The chosen seed is 64, corresponding to the maximum AUC<sub>-</sub>d<sub>e</sub>lt<sub>a pa</sub>r<sub>s</sub>im<sub>o</sub>ni<sub>ous</sub> m<sub>o</sub>d<sub>e</sub>l<sub>.</sub> M<sub>ea</sub>n AUC<sub>:</sub> 0<sub>.</sub>8321<sub>,</sub> M<sub>e</sub>di<sub>a</sub>n AUC<sub>:</sub> 0<sub>.</sub>8359

![](images/6f705bc5a5f59ea64bca1d948c26fd68eedda7e25cb2f704e712f94b33a9e076.jpg)

![](images/e329c3cfc828e5c03e00f8b4cb228e47d1e4e37fceb6e0fee591219c1b818801.jpg)  
Figure 17. The histogram of the Gower distance in test set. The chosen seed is 64, corresponding to the maximum AUC-delta <sub>pars</sub>i<sub>mon</sub>i<sub>ous</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> Th<sub>e</sub> hi<sub>s</sub>t<sub>ograms</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f G<sub>ower</sub> di<sub>s</sub>t<sub>ance</sub> i<sub>s</sub> <sub>c</sub>l<sub>ose</sub> t<sub>o</sub> P<sub>o</sub>i<sub>sson</sub> <sub>or</sub> G<sub>auss</sub>i<sub>an</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on,</sub> <sub>w</sub>hi<sub>c</sub>h h<sub>e</sub>l<sub>ps s</sub>h<sub>ow</sub> th<sub>a</sub>t b<sub>o</sub>th th<sub>e</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng se</sub>t <sub>an</sub>d t<sub>es</sub>ti<sub>ng se</sub>t <sub>are non-</sub>bi<sub>ase</sub>d <sub>an</sub>d <sub>are no</sub>t h<sub>av</sub>i<sub>ng pre</sub>f<sub>erence</sub> t<sub>owar</sub>d<sub>s spec</sub>ifi<sub>c su</sub>b<sub>groups w</sub>hi<sub>c</sub>h <sub>s</sub>h<sub>are cer</sub>t<sub>a</sub>i<sub>n s</sub>i<sub>m</sub>il<sub>ar</sub>iti<sub>es.</sub>

T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> B <sub>ase</sub>li<sub>ne</sub> D<sub>emograp</sub>hi<sub>c</sub> <sub>an</sub>d Cli<sub>n</sub>i<sub>ca</sub>l Ch<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub>
<table><tr><td colspan="5">Grouped by cvd_death</td></tr><tr><td></td><td>Missing</td><td>Overall</td><td>False</td><td>True</td><td>P-Value</td></tr><tr><td>n</td><td></td><td>17748</td><td>16992</td><td>756</td><td></td></tr><tr><td>Age (years), mean (SD)</td><td>0</td><td>51.5 (13.5)</td><td>50.9 (13.2)</td><td>65.3 (11.0)</td><td>&lt;0.001</td></tr><tr><td>Sex, n (%)</td><td>female</td><td>8989 (50.6)</td><td>8671 (51.0)</td><td>318 (42.1)</td><td>&lt;0.001</td></tr><tr><td></td><td>male</td><td>8759 (49.4)</td><td>8321 (49.0)</td><td>438 (57.9)</td><td></td></tr><tr><td>Race / Ethnicity, n (%)</td><td>Mexican</td><td>3682 (20.7)</td><td>3557 (20.9)</td><td>125 (16.5)</td><td>&lt;0.001</td></tr><tr><td></td><td>American</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Non-</td><td>3403 (19.2)</td><td>3222 (19.0)</td><td>181 (23.9)</td><td></td></tr><tr><td></td><td>Hispanic</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Black</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Non-</td><td>8719 (49.1)</td><td>8317 (48.9)</td><td>402 (53.2)</td><td></td></tr><tr><td></td><td>Hispanic</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>White</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Other 1</td><td>677 (3.8)</td><td>665 (3.9)</td><td>12 (1.6)</td><td></td></tr><tr><td></td><td>Multi-Racial</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Other His-</td><td>1267 (7.1)</td><td>1231 (7.2)</td><td>36 (4.8)</td><td></td></tr><tr><td></td><td>panic</td><td></td><td></td><td></td><td></td></tr><tr><td>Education Level, n (%)</td><td>Less than 9th</td><td>2449 (13.8)</td><td>2280 (13.4)</td><td>169 (22.4)</td><td>&lt;0.001</td></tr><tr><td></td><td>Grade</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Continued on next page</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> B <sub>ase</sub>li<sub>ne</sub> D<sub>emograp</sub>hi<sub>c</sub> <sub>an</sub>d Cli<sub>n</sub>i<sub>ca</sub>l Ch<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub>
<table><tr><td colspan="5"></td></tr><tr><td>9th-11th</td><td>Missing</td><td>Overall</td><td>False</td><td>True P-Value</td></tr><tr><td>Grade</td><td></td><td>2737 (15.4)</td><td>2581 (15.2)</td><td>156 (20.6)</td></tr><tr><td>High School Grad / GED</td><td></td><td>4142 (23.3)</td><td>3952 (23.3) 190 (25.1)</td><td></td></tr><tr><td>Some Col- lege / AA</td><td></td><td>4601 (25.9) 4462 (26.3)</td><td>139 (18.4)</td><td></td></tr><tr><td>degree College Graduate or</td><td></td><td>3798 (21.4)</td><td>3698 (21.8)</td><td>100 (13.2)</td></tr><tr><td>Above</td><td></td><td></td><td></td><td></td></tr><tr><td>Refused</td><td></td><td>8 (0.0)</td><td>6 (0.0)</td><td>2 (0.3)</td></tr><tr><td>Don&#x27;t Know</td><td></td><td>13 (0.1)</td><td>13 (0.1)</td><td>0 (0.0)</td></tr><tr><td>Poverty-Income Ratio, mean (SD)</td><td>1378</td><td>2.8 (1.6)</td><td>2.8 (1.6)</td><td>2.3 (1.5) &lt;0.001</td></tr><tr><td>Systolic BP (mmHg), mean</td><td>0</td><td>124.7 (17.3)</td><td></td><td>124.2 (17.0) 135.3 (19.3) &lt;0.001</td></tr><tr><td>(SD) Diastolic BP (mmHg), mean</td><td>0</td><td>72.2 (12.3)</td><td>72.3 (12.1)</td><td>69.1 (15.4) &lt;0.001</td></tr><tr><td>(SD)</td><td></td><td></td><td></td><td></td></tr><tr><td>Hypertension Meds, n (%) False</td><td></td><td>13467 (75.9) 13082 (77.0)</td><td></td><td>385 (50.9) &lt;0.001</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> B <sub>ase</sub>li<sub>ne</sub> D<sub>emograp</sub>hi<sub>c</sub> <sub>an</sub>d Cli<sub>n</sub>i<sub>ca</sub>l Ch<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub>
<table><tr><td colspan="6">Grouped by cvd_death</td></tr><tr><td></td><td>Missing</td><td>Overall</td><td>False</td><td>True</td><td>P-Value</td></tr><tr><td>True</td><td></td><td>4281 (24.1)</td><td>3910 (23.0)</td><td>371 (49.1)</td><td></td></tr><tr><td>Total Cholesterol (mg/dL), mean (SD)</td><td>0</td><td>204.4 (39.1)</td><td>204.4 (39.0)</td><td>204.7 (41.0)</td><td>0.846</td></tr><tr><td>HDL Cholesterol (mg/dL), mean (SD)</td><td>0</td><td>52.6 (15.6)</td><td>52.6 (15.6)</td><td>51.9 (15.7)</td><td>0.235</td></tr><tr><td>BMI (kg/m²), mean (SD)</td><td>0</td><td>28.7 (5.5)</td><td>28.7 (5.5)</td><td>29.2 (5.7)</td><td>0.035</td></tr><tr><td>Height (cm), mean (SD)</td><td>0</td><td>167.8 (10.1)</td><td>167.8 (10.1)</td><td>167.4 (10.1)</td><td>0.358</td></tr><tr><td>Waist Circumference (cm), mean (SD)</td><td>252</td><td>99.0 (15.1)</td><td>98.8 (15.0)</td><td>103.3 (15.8)</td><td>&lt;0.001</td></tr><tr><td>Diabetes Mellitus, n (%)</td><td></td><td>15887 (89.5)</td><td>15311 (90.1)</td><td>576 (76.2)</td><td>&lt;0.001</td></tr><tr><td></td><td>False True</td><td>1861 (10.5)</td><td>1681 (9.9)</td><td>180 (23.8)</td><td></td></tr><tr><td>Current Smoker, n (%)</td><td>False</td><td>13777 (77.6)</td><td>13210 (77.7)</td><td>567 (75.0)</td><td>0.084</td></tr><tr><td>True</td><td></td><td>3971 (22.4)</td><td>3782 (22.3)</td><td>189 (25.0)</td><td></td></tr><tr><td>Statin Use, n (%) False</td><td></td><td>15470 (87.2)</td><td></td><td></td><td>&lt;0.001</td></tr><tr><td>True</td><td></td><td>2278 (12.8)</td><td>14895 (87.7) 2097 (12.3)</td><td>575 (76.1) 181 (23.9)</td><td></td></tr><tr><td>Serum Creatinine (mg/dL),</td><td>0</td><td>0.9 (0.4)</td><td>0.9 (0.4)</td><td>1.0 (0.8)</td><td>&lt;0.001</td></tr><tr><td colspan="6">mean (SD)</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> B <sub>ase</sub>li<sub>ne</sub> D<sub>emograp</sub>hi<sub>c</sub> <sub>an</sub>d Cli<sub>n</sub>i<sub>ca</sub>l Ch<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub>
<table><tr><td colspan="2"></td><td colspan="4">Grouped by cvd_death</td></tr><tr><td></td><td>Missing</td><td>Overall</td><td>False</td><td>True</td><td>P-Value</td></tr><tr><td>eGFR (mL/min/1.73 m²), mean (SD)</td><td>0</td><td>94.6 (19.1)</td><td>95.2 (18.8)</td><td>81.4 (21.9)</td><td>&lt;0.001</td></tr><tr><td>HbA1c (%), median [Q1,Q3]</td><td>35</td><td>5.5 [5.2,5.8]</td><td>5.5 [5.2,5.8]</td><td>5.7 [5.4,6.3]</td><td>&lt;0.001</td></tr><tr><td>ALT (U/L), mean (SD)</td><td>43</td><td>26.9 (29.1)</td><td>27.0 (29.4)</td><td>24.6 (21.3)</td><td>0.003</td></tr><tr><td>AST (U/L), mean (SD)</td><td>43</td><td>26.3 (22.1)</td><td>26.3 (22.3)</td><td>26.5 (18.4)</td><td>0.842</td></tr><tr><td>Albumin (g/dL), mean (SD)</td><td>0</td><td>42.6 (3.2)</td><td>42.6 (3.2)</td><td>41.9 (3.2)</td><td>&lt;0.001</td></tr><tr><td>Hemoglobin (g/dL), mean</td><td>32</td><td>14.3 (1.5)</td><td>14.3 (1.5)</td><td>14.2 (1.6)</td><td>0.109</td></tr><tr><td>(SD) Hematocrit (%), mean (SD)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WBC Count (1000 cells/μL),</td><td>32 32</td><td>42.1 (4.3) 7.1 (2.4)</td><td>42.1 (4.3) 7.1 (2.4)</td><td>42.2 (4.5) 7.3 (2.2)</td><td>0.797 0.043</td></tr><tr><td>mean (SD) Platelet Count (1000 cells/μL),</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mean (SD)</td><td>32</td><td>264.0 (68.3)</td><td>264.4 (68.0)</td><td>254.2 (73.4)</td><td>&lt;0.001</td></tr><tr><td>CRP (mg/L), median [Q1,Q3]</td><td>6</td><td>2.1 [0.9,4.8]</td><td>2.1 [0.9,4.7]</td><td>3.0 [1.4,6.5]</td><td>&lt;0.001</td></tr><tr><td>History of Cancer, n (%) False</td><td></td><td>16270 (91.7)</td><td>15601 (91.8)</td><td>669 (88.5)</td><td>0.002</td></tr><tr><td>True</td><td></td><td>1478 (8.3)</td><td>1391 (8.2)</td><td>87 (11.5)</td><td></td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 3<sub>.</sub> B <sub>ase</sub>li<sub>ne</sub> D<sub>emograp</sub>hi<sub>c an</sub>d Cli<sub>n</sub>i<sub>ca</sub>l Ch<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub>
<table><tr><td colspan="2"></td><td rowspan="2">Overall</td><td colspan="3">Grouped by Outcome</td></tr><tr><td></td><td>Missing</td><td>High-Performance Subgroup</td><td>Low-Performance Subgroup</td><td>P-Value</td></tr><tr><td>n</td><td></td><td>3550</td><td>1775</td><td>1775</td><td></td></tr><tr><td>Hypertension Meds, n (%)</td><td>False</td><td>2698 (76.0)</td><td>1608 (90.6)</td><td>1090 (61.4)</td><td>&lt;0.001</td></tr><tr><td rowspan="2">Statin Use, n (%)</td><td>True</td><td>852 (24.0)</td><td>167 (9.4)</td><td>685 (38.6)</td><td></td></tr><tr><td>False</td><td>3099 (87.3)</td><td>1692 (95.3)</td><td>1407 (79.3)</td><td>&lt;0.001</td></tr><tr><td>Current Smoker, n (%)</td><td>True</td><td>451 (12.7)</td><td>83 (4.7)</td><td>368 (20.7)</td><td></td></tr><tr><td rowspan="2"></td><td>False</td><td>2786 (78.5)</td><td>1600 (90.1)</td><td>1186 (66.8)</td><td>&lt;0.001</td></tr><tr><td>True</td><td>764 (21.5)</td><td>175 (9.9)</td><td>589 (33.2)</td><td></td></tr><tr><td rowspan="2">Diabetes Mellitus, n (%)</td><td>False</td><td>3162 (89.1)</td><td>1710 (96.3)</td><td>1452 (81.8)</td><td>&lt;0.001</td></tr><tr><td>True</td><td>388 (10.9)</td><td>65 (3.7)</td><td>323 (18.2)</td><td></td></tr><tr><td>Age (years), mean (SD)</td><td>0</td><td>51.4 (13.5)</td><td>46.3 (12.2)</td><td>56.5 (12.9)</td><td>&lt;0.001</td></tr><tr><td>History of Cancer, n (%)</td><td>False</td><td>3291 (92.7) 259 (7.3)</td><td>1751 (98.6)</td><td>1540 (86.8)</td><td>&lt;0.001</td></tr><tr><td rowspan="4">Education Level, n (%)</td><td>True</td><td></td><td>24 (1.4)</td><td>235 (13.2)</td><td></td></tr><tr><td>Less than 9th</td><td>482 (13.6)</td><td>153 (8.6)</td><td>329 (18.5)</td><td>&lt;0.001</td></tr><tr><td>Grade</td><td>563 (15.9)</td><td></td><td></td><td></td></tr><tr><td>9th-11th Grade</td><td></td><td>180 (10.1)</td><td>383 (21.6)</td><td></td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 3<sub>.</sub> B <sub>ase</sub>li<sub>ne</sub> D<sub>emograp</sub>hi<sub>c an</sub>d Cli<sub>n</sub>i<sub>ca</sub>l Ch<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub>
<table><tr><td rowspan="2">Missing</td><td rowspan="2">Overall</td><td colspan="3">Grouped by Outcome</td></tr><tr><td>High-Performance Subgroup</td><td>Low-Performance Subgroup</td><td>P-Value</td></tr><tr><td>Grad / GED</td><td>High School</td><td>831 (23.4)</td><td>373 (21.0)</td><td>458 (25.8)</td></tr><tr><td>Some Col- lege / AA</td><td>887 (25.0)</td><td>496 (27.9)</td><td>391 (22.0)</td><td></td></tr><tr><td>degree College Graduate or</td><td>784 (22.1)</td><td>570 (32.1)</td><td>214 (12.1)</td><td></td></tr><tr><td>Above</td><td></td><td></td><td></td><td></td></tr><tr><td>Sex, n (%)</td><td>Don&#x27;t Know</td><td>3 (0.1) 1820 (51.3)</td><td>3 (0.2) 936 (52.7)</td><td>0 (0.0)</td></tr><tr><td>Female Male</td><td>1730 (48.7)</td><td>839 (47.3)</td><td>884 (49.8)</td><td>0.087</td></tr><tr><td>Poverty-Income Ratio, mean</td><td>0 2.7 (1.6)</td><td>3.3 (1.6)</td><td>891 (50.2)</td><td>&lt;0.001</td></tr><tr><td>(SD)</td><td></td><td></td><td></td><td>2.2 (1.5)</td></tr><tr><td>Systolic BP (mmHg), mean (SD)</td><td>0</td><td>124.5 (17.6)</td><td>119.3 (15.7)</td><td>129.8 (17.9) &lt;0.001</td></tr><tr><td></td><td></td><td></td><td></td><td>&lt;0.001</td></tr><tr><td>Race / Ethnicity, n (%) Mexican American</td><td></td><td>718 (20.2)</td><td>344 (19.4)</td><td>374 (21.1)</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 3<sub>.</sub> B <sub>ase</sub>li<sub>ne</sub> D<sub>emograp</sub>hi<sub>c an</sub>d Cli<sub>n</sub>i<sub>ca</sub>l Ch<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub>
<table><tr><td rowspan="2"></td><td rowspan="2">Missing Overall</td><td rowspan="2"></td><td colspan="3">Grouped by Outcome</td></tr><tr><td>High-Performance Subgroup</td><td>Low-Performance Subgroup</td><td>P-Value</td></tr><tr><td></td><td>Non- Hispanic</td><td>722 (20.3)</td><td>240 (13.5)</td><td>482 (27.2)</td><td></td></tr><tr><td></td><td>Black</td><td></td><td></td><td></td><td></td></tr><tr><td>Non-</td><td></td><td>1734 (48.8)</td><td>997 (56.2)</td><td>737 (41.5)</td><td></td></tr><tr><td>Hispanic White</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Other</td><td>1</td><td>128 (3.6)</td><td>41 (2.3)</td><td>87 (4.9)</td><td></td></tr><tr><td>Multi-Racial Other His-</td><td></td><td>248 (7.0)</td><td>153 (8.6)</td><td>95 (5.4)</td><td></td></tr><tr><td>panic Diastolic BP (mmHg), mean</td><td>0</td><td>72.0 (11.9)</td><td>71.4 (10.4)</td><td>72.6 (13.2)</td><td>0.003</td></tr><tr><td>(SD)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HDL Cholesterol (mg/dL), mean (SD)</td><td>0</td><td>52.8 (15.6)</td><td>52.6 (14.4)</td><td>53.1 (16.7)</td><td>0.382</td></tr><tr><td>Total Cholesterol (mg/dL), mean</td><td>0</td><td>205.1 (39.5)</td><td>205.4 (38.5)</td><td>204.7 (40.4)</td><td>0.592</td></tr><tr><td>(SD)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hemoglobin (g/dL), mean (SD)</td><td>0</td><td>14.3 (1.5)</td><td>14.3 (1.4)</td><td>14.3 (1.6)</td><td>0.909</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hematocrit (%), mean (SD)</td><td>0</td><td>42.1 (4.3)</td><td>42.0 (4.1)</td><td>42.1 (4.5)</td><td>0.577</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 3<sub>.</sub> B <sub>ase</sub>li<sub>ne</sub> D<sub>emograp</sub>hi<sub>c</sub> <sub>an</sub>d Cli<sub>n</sub>i<sub>ca</sub>l Ch<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs</sub>
<table><tr><td rowspan="2"></td><td rowspan="2">Missing</td><td rowspan="2">Overall</td><td colspan="3">Grouped by Outcome</td></tr><tr><td>High-Performance Subgroup</td><td>Low-Performance Subgroup</td><td>P-Value</td></tr><tr><td rowspan="2">cvd_death, n (%)</td><td>False</td><td></td><td>3399 (95.7)</td><td>1743 (98.2)</td><td></td><td></td></tr><tr><td>True</td><td></td><td>151 (4.3)</td><td>32 (1.8)</td><td>1656 (93.3) 119 (6.7)</td><td>&lt;0.001</td></tr><tr><td>weighted AUC, median [Q1,Q3]</td><td></td><td>0</td><td>0.8 [0.7,0.9]</td><td>0.9 [0.9,0.9]</td><td>0.7 [0.7,0.8]</td><td>&lt;0.001</td></tr></table>

## 10 REFERENCES

[1] P. A. Jaskowiak, I. G. Costa, and R. J. G. B. Cam<sub>p</sub>ello, “The area under the ROC curve as a measure of clusterin<sub>g</sub> <sub>q</sub>ualit<sub>y</sub>,” CoRR, vol. abs/2009.02400, 2020. arXiv: 2009.02400. [Online]. Available: https: //arxiv.org/abs/2009.02400

[2] S. S. Khan et al., “Develo<sub>p</sub>ment and validation of the american heart association’s <sub>p</sub>revent e<sub>q</sub>uations,” Circulation, vol. 149, no. 6, <sub>pp</sub>. 430–449, 2024. doi: 10.1161/CIRCULATIONAHA.123.067626 e<sub>p</sub>rint: https://www.ahajournals.org/doi/pdf/10.1161/CIRCULATIONAHA.123.067626. [Online]. Available: https://www.ahajournals.org/doi/abs/10.1161/CIRCULATIONAHA.123.067626

[3] S. workin<sub>g g</sub>rou<sub>p</sub> and E. C. risk collaboration, “Score2 risk <sub>p</sub>rediction al<sub>g</sub>orithms: New models to estimate 10-year risk of cardiovascular disease in europe,” European Heart Journal, vol. 42, no. 25, pp. 2439– 2454, Jul. 2021, issn: 0195-668X. doi: 10.1093/eurheartj/ehab309 e<sub>p</sub>rint: https://academic. oup.com/eurheartj/article-pdf/42/25/2439/46633454/ehab309.pdf. [Online]. Available: https://doi.org/10.1093/eurheartj/ehab309

[4] R. Mondal and R. Ritu, “Aha’s new <sub>p</sub>revent 10-<sub>y</sub>ear cardiovascular risk and all-cause, cvd, and non-cvd mortality among us adults: The nhanes 2003–2019,” Hipertension y Riesgo Vascular ´ , vol. 42, no. 4, pp. 259– 266, 2025, issn: 1889-1837. doi: https://doi.org/10.1016/j.hipert.2025.02.003 [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1889183725000455

[5] X. Yan et al., “Performance of the american heart association’s <sub>p</sub>revent e<sub>q</sub>uations amon<sub>g</sub> disa<sub>gg</sub>re<sub>g</sub>ated racial and ethnic subgroups,” JAMA Cardiology, vol. 10, no. 9, pp. 876–885, Sep. 2025, issn: 2380-6583. doi: 10. 1001/jamacardio.2025.1865 e<sub>p</sub>rint: https://jamanetwork.com/journals/jamacardiology/ articlepdf/2835783/jamacardiology\_yan\_2025\_oi\_250031\_1757354399.78721.pdf. [Online]. Available: https://doi.org/10.1001/jamacardio.2025.1865

[6] S. V. Rodne<sub>y</sub> A Ha<sub>y</sub>ward David M Kent and T. P. Hofer, “Multivariable risk <sub>p</sub>rediction can <sub>g</sub>reatl<sub>y</sub> enhance the statistical power of clinical trial subgroup analysis,” BMC Med Res Methodol, vol. 6, no. 18, 2006. doi: https://doi.org/10.1186/1471-2288-6-18

[7] E. K. Oikonomou, E. S. S<sub>p</sub>atz, M. A. Suchard, and R. Khera, “Individualisin<sub>g</sub> intensive s<sub>y</sub>stolic blood <sub>pressure re</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>n</sub> h<sub>yper</sub>t<sub>ens</sub>i<sub>on us</sub>i<sub>ng compu</sub>t<sub>a</sub>ti<sub>ona</sub>l t<sub>r</sub>i<sub>a</sub>l <sub>p</sub>h<sub>enomaps an</sub>d <sub>mac</sub>hi<sub>ne</sub> l<sub>earn</sub>i<sub>ng:</sub> A <sub>pos</sub>t hoc analysis of randomised clinical trials,” The Lancet Digital Health, vol. 4, no. 11, e796–e805, 2022, issn: 2589-7500. doi: https://doi.org/10.1016/S2589-7500(22)00170-4 [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2589750022001704

[8] J. C. Gower, “A general coeficient of similarity and some of its properties,” Biometrics, vol. 27, no. 4, <sub>pp</sub>. 857–871, 1971, issn: 0006341X, 15410420. Accessed: Jul. 10, 2026. [Online]. Available: http : //www.jstor.org/stable/2528823

[9] L. A. C. Millard, P. A. Flach, and J. P. T. Hi<sub>gg</sub>ins, “Rate-constrained rankin<sub>g</sub> and the rate-wei<sub>g</sub>hted auc,” in Machine Learning and Knowledge Discovery in Databases, T. Calders, F. Esposito, E. H¨ullermeier, and R M<sub>eo,</sub> Ed<sub>s.,</sub> B<sub>er</sub>li<sub>n,</sub> H<sub>e</sub>id<sub>e</sub>lb<sub>erg:</sub> S<sub>pr</sub>i<sub>nger</sub> B<sub>er</sub>li<sub>n</sub> H<sub>e</sub>id<sub>e</sub>lb<sub>erg,</sub> 2014<sub>,</sub> <sub>pp.</sub> 386<sub>–</sub>403<sub>,</sub> <sub>isbn:</sub> 978<sub>-</sub>3<sub>-</sub>662<sub>-</sub>44851<sub>-</sub>9<sub>.</sub>

[10] B. Rozemberczki et al., The shapley value in machine learning, 2022. arXiv: 2202 . 05594 [cs.LG]. [Online]. Available: https://arxiv.org/abs/2202.05594

[11] T. Fawcett, “An introduction to roc analysis,” Pattern Recognition Letters, vol. 27, no. 8, pp. 861–874, 2006, ROC Anal<sub>y</sub>sis in Pattern Reco<sub>g</sub>nition, issn: 0167-8655. doi: https://doi.org/10.1016/j.patrec. 2005.10.010 [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S016786550500303X

[12] Centers for Disease Control and Prevention. “About the national health and nutrition examination surve<sub>y</sub>,” Accessed: Jul. 31, 2026. [Online]. Available: https://www.cdc.gov/nchs/nhanes/about/index. html

[13] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, <sub>vo</sub>l<sub>.</sub> 12<sub>,</sub> <sub>pp.</sub> 2825<sub>–</sub>2830<sub>,</sub> 2011<sub>.</sub>

[14] M. Mayer, Preventr: An implementation of the prevent and pooled cohort equations, R package version 0.12.0, 2026. [Online]. Available: https://martingmayer.com/preventr

[15] G. Biau and B. Cadre, “Optimization by gradient boosting,” in Advances in Contemporary Statistics and Econometrics: Festschrift in Honor ofChristine Thomas-Agnan, A. Daouia and A. Ruiz-Gazen, Eds. Cham: S<sub>p</sub>rin<sub>g</sub>er International Publishin<sub>g</sub>, 2021, <sub>pp</sub>. 23–44, isbn: 978-3-030-73249-3. doi: 10.1007/978-3-030- 73249-3\_2 [Online]. Available: https://doi.org/10.1007/978-3-030-73249-3\_2

[16] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” CoRR, vol. abs/1603.02754, 2016. arXiv: 1603.02754. [Online]. Available: http://arxiv.org/abs/1603.02754

[17] J. H. Friedman, “Greedy function approximation: A gradient boosting machine.,” Annals of Statistics, vol. 29, <sub>pp</sub>. 1189–1232, 2001. [Online]. Available: https://api.semanticscholar.org/CorpusID:39450643

[18] A. J. Lee, U-Statistics: Theory and Practice (Statistics: A Series of Textbooks and Monographs). CRC Press, 1990, vol. 110, isbn: 9780824782535. doi: 10.1201/9780203734520 [Online]. Available: https: //doi.org/10.1201/9780203734520

[19] E. Slutsky, “Uber stochastische asymptoten und grenzwerte,” <sup>¨</sup> Metron (in German), vol. 5, no. 3, pp. 3–89, 1925<sub>.</sub>

[20] L. McInnes, J. Healy, and J. Melville, Umap: Uniform manifold approximation and projection for dimension reduction, 2020. arXiv: 1802.03426 [stat.ML]. [Online]. Available: https://arxiv.org/abs/1802. 03426