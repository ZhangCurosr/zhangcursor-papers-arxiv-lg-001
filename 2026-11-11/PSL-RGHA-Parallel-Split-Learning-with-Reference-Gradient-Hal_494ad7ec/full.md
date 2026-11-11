# PSL-RGHA: Parallel Split Learning with Reference Gradient Half-Space Alignment for Mitigating Gradient Divergence

H<sub>a</sub>n Li<sup>∗</sup>   
18850543060@163.com   
Xi<sub>a</sub>m<sub>e</sub>n Uni<sub>ve</sub>r<sub>s</sub>it<sub>y o</sub>f Technolo<sub>gy</sub>   
Xiamen, Fujian, C<sup>h</sup>ina   
Jin<sub>g</sub> Yan<sub>g</sub>   
s2147529@siswa.um.edu.m<sub>y</sub>   
Uni<sub>ve</sub>r<sub>s</sub>iti M<sub>a</sub>l<sub>aya</sub>   
K<sub>u</sub>ala L<sub>u</sub>m<sub>pu</sub>r<sub>,</sub> Mala<sub>y</sub>sia Jianmin Li <sup>l</sup>ijm@xmut.e<sup>d</sup>u.cn   
Xi<sub>a</sub>m<sub>e</sub>n Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Technolo<sub>gy</sub>   
Xiamen, Fujian, C<sup>h</sup>ina

Zh<sub>e</sub>n<sub>g</sub> Lin<sup>∗</sup> z<sup>h</sup>en<sub>g</sub><sup>l</sup>in@ieee.or<sub>g</sub> SnT<sub>,</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Luxembour<sub>g</sub> Luxembour<sub>g</sub>

Zih<sub>a</sub>n F<sub>a</sub>n<sub>g</sub>   
<sub>z</sub>ih<sub>an</sub>f<sub>ang</sub>3<sub>-</sub>   
<sub>c@my.c</sub>it<sub>yu.e</sub>d<sub>u.</sub>hk   
Cit<sub>y</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f H<sub>o</sub>n<sub>g</sub>   
<sup>K</sup>on<sub>g</sub>   
Hon<sub>g</sub> Kon<sub>g,</sub> China Lin Li li<sub>n</sub>li<sub>@xmu</sub>t<sub>.e</sub>d<sub>u.cn</sub>   
Xi<sub>a</sub>m<sub>e</sub>n Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Technolo<sub>gy</sub>   
Xiamen, Fujian, C<sup>h</sup>ina Mi<sub>ao</sub> Y<sub>a</sub>n<sub>g</sub><sup>✉</sup>   
<sub>y</sub>an<sub>g</sub>miao@xmut.e<sup>d</sup>u.cn   
Xi<sub>a</sub>m<sub>e</sub>n Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Technolo<sub>gy</sub>   
Xiamen, Fujian, C<sup>h</sup>ina   
Qiyang Zhang   
<sub>q</sub>i<sub>y</sub>an<sub>g</sub>zhan<sub>g</sub>@<sub>p</sub>ku.edu.cn   
P<sub>e</sub>kin<sub>g</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub>   
Beijing, C<sup>h</sup>ina   
Min<sub>gwe</sub>i H<sub>o</sub>n<sub>g</sub>   
mw<sup>h</sup>on<sub>g</sub>12345@<sub>g</sub>mai<sup>l</sup>.com   
Xi<sub>a</sub>m<sub>e</sub>n Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f   
Technolo<sub>gy</sub>   
Xiamen, Fujian, C<sup>h</sup>ina Z<sub>e</sub>h<sub>a</sub>n<sub>g</sub> Lin   
18850975878@163.com   
Xi<sub>a</sub>m<sub>e</sub>n Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Technolo<sub>gy</sub>   
Xiamen, Fujian, C<sup>h</sup>ina   
Zhaolu Kan<sub>g</sub>   
kan<sub>g</sub>zl9966@<sub>g</sub>mail.com   
P<sub>e</sub>kin<sub>g</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub>   
Beijing, C<sup>h</sup>ina

Sh<sub>unz</sub>hi Zh<sub>u</sub><sup>✉</sup> <sub>szz</sub>h<sub>u@xmu</sub>t<sub>.e</sub>d<sub>u.cn</sub> Xi<sub>a</sub>m<sub>e</sub>n Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Technolo<sub>gy</sub> Xiamen, Fujian, C<sup>h</sup>ina

## Abstract

Th<sub>e esca</sub>l<sub>a</sub>ti<sub>ng comp</sub>l<sub>ex</sub>it<sub>y o</sub>f d<sub>eep neura</sub>l <sub>ne</sub>t<sub>wor</sub>k<sub>s</sub> hi<sub>n</sub>d<sub>ers</sub> th<sub>e</sub> de<sub>p</sub>lo<sub>y</sub>ment of federated learnin<sub>g</sub> (FL) on resource-constrained devices<sub>,</sub> es<sub>p</sub>eciall<sub>y</sub> for multimedia a<sub>pp</sub>lications such as on-device ima<sub>g</sub>e classification and s<sub>p</sub>eech reco<sub>g</sub>nition. Parallel s<sub>p</sub>lit learn in<sub>g</sub> (PSL) has emer<sub>g</sub>ed as a <sub>p</sub>romisin<sub>g</sub> solution b<sub>y</sub> ofloadin<sub>g</sub> a <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>por</sub>ti<sub>on o</sub>f th<sub>e compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>wor</sub>kl<sub>oa</sub>d t<sub>o</sub> th<sub>e server</sub> throu<sub>g</sub>h la<sub>y</sub>er-wise model <sub>p</sub>artitionin<sub>g,</sub> makin<sub>g</sub> it <sub>p</sub>articularl<sub>y</sub> suit <sub>a</sub>bl<sub>e</sub> f<sub>or resource-</sub>li<sub>m</sub>it<sub>e</sub>d <sub>v</sub>i<sub>sua</sub>l <sub>an</sub>d <sub>speec</sub>h i<sub>n</sub>t<sub>e</sub>lli<sub>gence scenar</sub>i<sub>os.</sub> However<sub>,</sub> the absence of client-side a<sub>gg</sub>re<sub>g</sub>ation renders PSL susce<sub>p</sub>tible to severe trainin<sub>g</sub> diver<sub>g</sub>ence and <sub>p</sub>otential conver<sub>g</sub>ence f<sub>a</sub>il<sub>ure un</sub>d<sub>er</sub> h<sub>e</sub>t<sub>erogeneous</sub> d<sub>a</sub>t<sub>a</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons.</sub> T<sub>o a</sub>dd<sub>ress</sub> thi<sub>s c</sub>h<sub>a</sub>l<sub>-</sub> l<sub>e</sub>n<sub>ge, we p</sub>r<sub>opose</sub> PSL-RGHA<sub>, a g</sub>r<sub>a</sub>di<sub>e</sub>nt-<sub>a</sub>li<sub>g</sub>n<sub>e</sub>d PSL fr<sub>a</sub>m<sub>ewo</sub>rk th<sub>a</sub>t <sub>emp</sub>l<sub>oys re</sub>f<sub>erence gra</sub>di<sub>en</sub>t h<sub>a</sub>lf<sub>-space a</sub>li<sub>gnmen</sub>t <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> t<sub>o</sub> miti<sub>ga</sub>t<sub>e</sub> tr<sub>a</sub>inin<sub>g</sub> di<sub>ve</sub>r<sub>ge</sub>n<sub>ce a</sub>m<sub>o</sub>n<sub>g c</sub>li<sub>e</sub>nt d<sub>ev</sub>i<sub>ces.</sub> PSL-RGHA com<sub>p</sub>rises two novel com<sub>p</sub>onents: reference <sub>g</sub>radient <sub>p</sub>ro<sub>g</sub>ressive construction (RGPC) and <sub>g</sub>radient half-s<sub>p</sub>ace ali<sub>g</sub>nment (GHA). S<sub>p</sub>ecificall<sub>y,</sub> RGPC e<sub>v</sub>al<sub>u</sub>ates the directional consistenc<sub>y</sub> of client <sub>g</sub>radients and <sub>p</sub>ro<sub>g</sub>ressivel<sub>y</sub> s<sub>y</sub>nthesizes a robust reference <sub>g</sub>radient via Weiszfeld u<sub>p</sub>dates<sub>,</sub> efectivel<sub>y</sub> a<sub>pp</sub>roximatin<sub>g</sub> the <sub>g</sub>lobal descent dir<sub>ec</sub>ti<sub>o</sub>n<sub>.</sub> Th<sub>e</sub>n<sub>,</sub> GHA <sub>co</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>s</sub> <sub>c</sub>li<sub>e</sub>nt <sub>g</sub>r<sub>a</sub>di<sub>e</sub>nt<sub>s</sub> <sub>w</sub>ithin th<sub>e</sub> r<sub>e</sub>f<sub>e</sub>r-<sub>ence</sub> h<sub>a</sub>lf<sub>-space</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> RGPC t<sub>o m</sub>iti<sub>ga</sub>t<sub>e gra</sub>di<sub>en</sub>t <sub>con</sub>fli<sub>c</sub>t<sub>s an</sub>d rectif<sub>y</sub> local u<sub>p</sub>dates. Extensive ex<sub>p</sub>eriments demonstrate that our

PSL-RGHA fr<sub>a</sub>m<sub>ewo</sub>rk <sub>ou</sub>t<sub>pe</sub>rf<sub>o</sub>rm<sub>s e</sub>xi<sub>s</sub>tin<sub>g</sub> b<sub>ase</sub>lin<sub>es</sub> in <sub>va</sub>ri<sub>ous</sub> sett<sup>i</sup>n<sub>g</sub>s.

CCS Concepts

• Computing methodologies → Machine learning; Distributed computing methodologies.

## Keywords

Distributed learnin<sub>g;</sub> s<sub>p</sub>lit learnin<sub>g;</sub> <sub>g</sub>radient ali<sub>g</sub>nment<sub>;</sub> multimedia intelli<sub>g</sub>ence

ACM Reference Format:   
Han Li, Zheng Lin, Miao Yang, Zehang Lin, Jing Yang, Zihan Fang, Qiyang Zhan<sub>g</sub>, Zhaolu Kan<sub>g</sub>, Jianmin Li, Lin Li, Min<sub>g</sub>wei Hon<sub>g</sub>, and Shunzhi Zhu. 2026<sub>.</sub> PSL-RGHA: P<sub>a</sub>r<sub>a</sub>ll<sub>e</sub>l S<sub>p</sub>lit L<sub>ea</sub>rnin<sub>g w</sub>ith R<sub>e</sub>f<sub>e</sub>r<sub>e</sub>n<sub>ce</sub> Gr<sub>a</sub>di<sub>e</sub>nt H<sub>a</sub>lf-Space Alignment for Mitigating Gradient Divergence. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. htt<sub>p</sub>s://doi.or<sub>g</sub>/10.1145/3767308.3835121

## 1 Introduction

With the <sub>p</sub>roliferation of the Internet of Thin<sub>g</sub>s (IoT), billions of interconnected smart devices continuousl<sub>y g</sub>enerate massive <sub>p</sub>rivac<sub>y</sub>- sensitive data [33, 49]. This un<sub>p</sub>recedented volume of data has si<sub>g</sub>nificantl<sub>y</sub> advanced the maturit<sub>y</sub> of machine learnin<sub>g</sub> (ML), en ablin<sub>g</sub> intelli<sub>g</sub>ent IoT a<sub>pp</sub>lications [8, 21, 34, 35, 56]. In <sub>p</sub>articular, th<sub>e rap</sub>id <sub>grow</sub>th <sub>o</sub>f <sub>mu</sub>lti<sub>me</sub>di<sub>a</sub> i<sub>n</sub>t<sub>e</sub>lli<sub>gence, w</sub>hi<sub>c</sub>h <sub>encompasses</sub> hi<sub>g</sub>h-fidelit<sub>y</sub> ima<sub>g</sub>e reco<sub>g</sub>nition and real-time s<sub>p</sub>eech <sub>p</sub>rocessin<sub>g,</sub> h<sub>as</sub> i<sub>mpose</sub>d <sub>ex</sub>t<sub>reme</sub> d<sub>eman</sub>d<sub>s on</sub> th<sub>e compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>an</sub>d <sub>memory</sub> ca<sub>p</sub>abilities of ed<sub>g</sub>e devices [2, 22, 43]. Conventional ML, known as centralized learnin<sub>g</sub> (CL), relies on a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> raw data at a <sub>cen</sub>t<sub>ra</sub>l <sub>server</sub> f<sub>or</sub> <sub>mo</sub>d<sub>e</sub>l t<sub>ra</sub>i<sub>n</sub>i<sub>ng.</sub> H<sub>owever,</sub> <sub>w</sub>ith th<sub>e</sub> <sub>exponen</sub>ti<sub>a</sub>l <sub>grow</sub>th <sub>o</sub>f <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l d<sub>a</sub>t<sub>a</sub> <sub>crea</sub>ti<sub>on,</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>a</sub>t 173<sub>.</sub>4 <sub>ze</sub>tt<sub>a</sub>b<sub>y</sub>t<sub>es</sub> i<sub>n</sub>

2025 and forecasted to tri<sub>p</sub>le to 527.5 zettab<sub>y</sub>tes b<sub>y</sub> 2029 [28, 38], CL encounters severe bottlenecks re<sub>g</sub>ardin<sub>g</sub> communication overhead and <sub>p</sub>rivac<sub>y</sub> leaka<sub>g</sub>e [4, 48]. To address these issues, federated learnin<sub>g</sub> (FL) [7, 36, 52] has emer<sub>g</sub>ed as a <sub>p</sub>rivac<sub>y</sub>-<sub>p</sub>reservin<sub>g</sub> <sub>p</sub>aradi<sub>gm, w</sub>hi<sub>c</sub>h <sub>ena</sub>bl<sub>es co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>ve</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> b<sub>y exc</sub>h<sub>ang</sub>i<sub>ng mo</sub>d<sub>e</sub>l u<sub>p</sub>dates rather than raw data. While FL efectivel<sub>y</sub> miti<sub>g</sub>ates data <sub>p</sub>rivac<sub>y</sub> and communication concerns<sub>,</sub> it confronts severe scalabilit<sub>y</sub> b<sub>o</sub>ttl<sub>enec</sub>k<sub>s</sub> <sub>as</sub> <sub>mo</sub>d<sub>ern</sub> ML <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>expan</sub>d <sub>exponen</sub>ti<sub>a</sub>ll<sub>y</sub> i<sub>n</sub> size [9, 18, 27, 30, 54].

To overcome these bottlenecks, s<sub>p</sub>lit learnin<sub>g</sub> (SL) [26, 40] has <sub>emerge</sub>d t<sub>o</sub> <sub>par</sub>titi<sub>on</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l l<sub>ayer-w</sub>i<sub>se,</sub> <sub>o</sub>fl<sub>oa</sub>di<sub>ng</sub> th<sub>e</sub> <sub>compu</sub>t<sub>e-</sub> intensive workload to ed<sub>g</sub>e server. However<sub>,</sub> vanilla SL ado<sub>p</sub>ts se-<sub>q</sub>uential trainin<sub>g p</sub>i<sub>p</sub>eline<sub>,</sub> which inherentl<sub>y</sub> sufers from exces-<sub>s</sub>i<sub>ve</sub> l<sub>a</sub>t<sub>ency</sub> <sub>an</sub>d <sub>sys</sub>t<sub>em</sub> <sub>over</sub>h<sub>ea</sub>d d<sub>ue</sub> t<sub>o</sub> th<sub>e</sub> idl<sub>e</sub> <sub>wa</sub>iti<sub>ng</sub> ti<sub>me</sub> on clients [10]. To address this limitation, s<sub>p</sub>lit federated learnin<sub>g</sub> (SFL) [29, 31, 39] has been <sub>p</sub>ro<sub>p</sub>osed b<sub>y</sub> <sub>p</sub>arallelizin<sub>g</sub> client trainin<sub>g</sub>, <sub>y</sub>et it reintroduces a si<sub>g</sub>nificant communication burden b<sub>y</sub> re<sub>q</sub>uirin<sub>g</sub> <sub>c</sub>li<sub>en</sub>t<sub>s</sub> t<sub>o</sub> <sub>per</sub>i<sub>o</sub>di<sub>ca</sub>ll<sub>y</sub> <sub>sync</sub>h<sub>ron</sub>i<sub>ze</sub> th<sub>e</sub>i<sub>r</sub> l<sub>oca</sub>l <sub>su</sub>b<sub>-mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>w</sub>ith <sub>a</sub> <sub>pa-</sub> rameter server. Conse<sub>q</sub>uentl<sub>y</sub>, <sub>p</sub>arallel s<sub>p</sub>lit learnin<sub>g</sub> (PSL) [15, 17] was <sub>p</sub>ro<sub>p</sub>osed as a hi<sub>g</sub>hl<sub>y</sub> streamlined architecture. B<sub>y</sub> com<sub>p</sub>letel<sub>y</sub> eliminatin<sub>g</sub> client-side <sub>p</sub>arameter a<sub>gg</sub>re<sub>g</sub>ation<sub>,</sub> PSL successfull<sub>y</sub> reduces communication overhead and s<sub>y</sub>stem com<sub>p</sub>lexit<sub>y,</sub> makin<sub>g</sub> it the most resource-eficient <sub>p</sub>aradi<sub>g</sub>m for de<sub>p</sub>lo<sub>y</sub>in<sub>g</sub> lar<sub>g</sub>e-scale <sub>mo</sub>d<sub>e</sub>l<sub>s on</sub> th<sub>e e</sub>d<sub>ge.</sub>

H<sub>oweve</sub>r<sub>,</sub> PSL intr<sub>o</sub>d<sub>uces seve</sub>r<sub>e</sub> tr<sub>a</sub>inin<sub>g</sub> in<sub>s</sub>t<sub>a</sub>bilit<sub>y</sub> in h<sub>e</sub>t<sub>e</sub>r<sub>o</sub>- <sub>ge</sub>n<sub>eous e</sub>n<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt<sub>s.</sub> In PSL<sub>, eac</sub>h <sub>c</sub>li<sub>e</sub>nt <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>es</sub> int<sub>e</sub>rm<sub>e</sub>di<sub>a</sub>t<sub>e</sub> activations based solel<sub>y</sub> on its local<sub>,</sub> t<sub>yp</sub>icall<sub>y</sub> non-IID data distri bution [45, 59]. In the absence of client-side s<sub>y</sub>nchronization, each <sub>c</sub>li<sub>en</sub>t’<sub>s su</sub>b<sub>-mo</sub>d<sub>e</sub>l <sub>rap</sub>idl<sub>y over</sub>fit<sub>s</sub> t<sub>o</sub> it<sub>s</sub> l<sub>oca</sub>l d<sub>a</sub>t<sub>a</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> Thi<sub>s</sub> l<sub>oca</sub>l di<sub>vergence</sub> <sub>causes</sub> <sub>c</sub>li<sub>en</sub>t<sub>s</sub> t<sub>o</sub> <sub>up</sub>l<sub>oa</sub>d <sub>seman</sub>ti<sub>ca</sub>ll<sub>y</sub> i<sub>ncons</sub>i<sub>s</sub>t<sub>en</sub>t activations to the server [6, 25]. Conse<sub>q</sub>uentl<sub>y</sub>, durin<sub>g</sub> server-side b<sub>ac</sub>k<sub>propaga</sub>ti<sub>on,</sub> th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>l <sub>rece</sub>i<sub>ves</sub> <sub>w</sub>ildl<sub>y</sub> di<sub>vergen</sub>t <sub>an</sub>d conflictin<sub>g</sub> <sub>g</sub>radients. Rather than conver<sub>g</sub>in<sub>g,</sub> the accumulation of th<sub>ese</sub> <sub>con</sub>fli<sub>c</sub>ti<sub>ng</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>causes</sub> d<sub>es</sub>t<sub>ruc</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>er</sub>f<sub>erence,</sub> l<sub>ea</sub>di<sub>ng</sub> t<sub>o</sub> severe client drift and <sub>p</sub>otential conver<sub>g</sub>ence failure. Since reintroducin<sub>g</sub> client-side a<sub>gg</sub>re<sub>g</sub>ation would destro<sub>y</sub> PSL’s communication <sub>e</sub>fi<sub>c</sub>i<sub>ency,</sub> th<sub>e cr</sub>iti<sub>ca</sub>l <sub>c</sub>h<sub>a</sub>ll<sub>enge</sub> i<sub>s</sub> h<sub>ow</sub> t<sub>o reso</sub>l<sub>ve</sub> th<sub>ese gra</sub>di<sub>en</sub>t <sub>con</sub>fli<sub>c</sub>t<sub>s</sub> <sub>en</sub>ti<sub>re</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>server.</sub>

To address the trainin<sub>g</sub> instabilit<sub>y</sub> inherent in PSL<sub>, w</sub>e <sub>p</sub>ro<sub>p</sub>ose PSL-RGHA<sub>, a g</sub>r<sub>a</sub>di<sub>e</sub>nt-<sub>a</sub>li<sub>g</sub>n<sub>e</sub>d PSL fr<sub>a</sub>m<sub>ewo</sub>rk th<sub>a</sub>t <sub>e</sub>m<sub>p</sub>l<sub>oys a</sub> reference <sub>g</sub>radient half-s<sub>p</sub>ace ali<sub>g</sub>nment mechanism to miti<sub>g</sub>ate conflictin<sub>g</sub> <sub>up</sub>dates amon<sub>g</sub> client de<sub>v</sub>ices. PSL-RGHA com<sub>p</sub>rises two s<sub>y</sub>ner<sub>g</sub>istic com<sub>p</sub>onents desi<sub>g</sub>ned to enforce <sub>g</sub>lobal consistenc<sub>y</sub> without client-side a<sub>gg</sub>re<sub>g</sub>ation. S<sub>p</sub>ecificall<sub>y,</sub> to a<sub>pp</sub>roximate the <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l d<sub>escen</sub>t di<sub>rec</sub>ti<sub>on,</sub> <sub>we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>re</sub>f<sub>erence</sub> <sub>gra</sub>di<sub>en</sub>t <sub>progres-</sub> sive construction (RGPC), which <sub>p</sub>ro<sub>g</sub>ressivel<sub>y</sub> filters and s<sub>y</sub>nthesizes directionall<sub>y</sub> consistent client <sub>g</sub>radients into a robust reference vector. Buildin<sub>g</sub> u<sub>p</sub>on this trusted reference<sub>,</sub> we devise a <sub>g</sub>radi ent half-s<sub>p</sub>ace ali<sub>g</sub>nment (GHA) scheme that <sub>p</sub>rojects conflictin<sub>g</sub> <sub>c</sub>li<sub>en</sub>t <sub>gra</sub>di<sub>en</sub>t<sub>s on</sub>t<sub>o</sub> th<sub>e non-nega</sub>ti<sub>ve</sub> h<sub>a</sub>lf<sub>-space o</sub>f th<sub>e re</sub>f<sub>erence</sub> direction<sub>,</sub> efectivel<sub>y</sub> neutralizin<sub>g</sub> com<sub>p</sub>onents that im<sub>p</sub>ede con-<sub>vergence.</sub> Thi<sub>s</sub> <sub>co</sub>ll<sub>a</sub>b<sub>ora</sub>ti<sub>on</sub> <sub>ensures</sub> th<sub>a</sub>t l<sub>oca</sub>l <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>are</sub> <sub>con-</sub> ti<sub>nuous</sub>l<sub>y</sub> h<sub>armon</sub>i<sub>ze</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>convergence</sub> t<sub>ren</sub>d <sub>w</sub>ith<sub>ou</sub>t re<sub>q</sub>uirin<sub>g</sub> client-side a<sub>gg</sub>re<sub>g</sub>ation. The main contributions of this <sub>paper are summar</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

• We <sub>p</sub>ro<sub>p</sub>ose PSL-RGHA, a novel <sub>g</sub>radient ali<sub>g</sub>nment framework for PSL that ex<sub>p</sub>licitl<sub>y</sub> miti<sub>g</sub>ates <sub>g</sub>radient conflicts in hetero<sub>g</sub>eneous data environments via <sub>p</sub>ure server-side <sub>g</sub>eometric ali<sub>g</sub>nment.

• We develo<sub>p</sub> the RGPC al<sub>g</sub>orithm, which utilizes directional consistenc<sub>y</sub> metrics and Weiszfeld u<sub>p</sub>dates to construct a noise-resilient reference <sub>g</sub>radient that accuratel<sub>y</sub> ca<sub>p</sub>tures th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>convergence</sub> t<sub>ren</sub>d<sub>.</sub>

• We devise the GHA mechanism, which <sub>p</sub>roject client <sub>g</sub>radi-<sub>en</sub>t<sub>s</sub> <sub>on</sub>t<sub>o</sub> th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> h<sub>a</sub>lf<sub>-space,</sub> th<sub>ere</sub>b<sub>y</sub> <sub>rec</sub>tif<sub>y</sub>i<sub>ng</sub> di<sub>ver-</sub> <sub>g</sub>ent u<sub>p</sub>dates and harmonizin<sub>g</sub> the o<sub>p</sub>timization direction <sub>across</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d <sub>c</sub>li<sub>en</sub>t<sub>s.</sub>

• We conduct extensive ex<sub>p</sub>eriments to evaluate PSL-RGHA. E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>resu</sub>lt<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t <sub>our</sub> f<sub>ramewor</sub>k <sub>s</sub>i<sub>gn</sub>ifi<sub>-</sub> cantl<sub>y</sub> out<sub>p</sub>erforms existin<sub>g</sub> baselines in both conver<sub>g</sub>ence <sub>spee</sub>d <sub>an</sub>d fi<sub>na</sub>l <sub>accuracy</sub> <sub>across</sub> di<sub>verse</sub> d<sub>a</sub>t<sub>a</sub> <sub>se</sub>tti<sub>ngs.</sub>

## 2 Related Work

## 2.1 Split Learning and Variants

SL was originally introduced by Vepakomma et al. [40] as a privacy-<sub>p</sub>reservin<sub>g</sub> alternative to FL. In vanilla SL framework<sub>,</sub> the <sub>g</sub>lobal <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> <sub>par</sub>titi<sub>one</sub>d i<sub>n</sub>t<sub>o</sub> <sub>c</sub>li<sub>en</sub>t<sub>-s</sub>id<sub>e</sub> <sub>an</sub>d <sub>server-s</sub>id<sub>e</sub> <sub>su</sub>b<sub>-mo</sub>d<sub>e</sub>l<sub>s.</sub> Throu<sub>g</sub>hout trainin<sub>g,</sub> clients interact with the server se<sub>q</sub>uentiall<sub>y</sub> i<sub>n a roun</sub>d<sub>-ro</sub>bi<sub>n</sub> f<sub>as</sub>hi<sub>on</sub> t<sub>o compu</sub>t<sub>e</sub> f<sub>orwar</sub>d <sub>an</sub>d b<sub>ac</sub>k<sub>war</sub>d <sub>passes.</sub> H<sub>oweve</sub>r<sub>,</sub> thi<sub>s seque</sub>nti<sub>a</sub>l tr<sub>a</sub>inin<sub>g</sub> in<sub>cu</sub>r<sub>s seve</sub>r<sub>e</sub> tr<sub>a</sub>inin<sub>g</sub> l<sub>a</sub>t<sub>e</sub>n<sub>cy,</sub> hi<sub>n</sub>d<sub>er</sub>i<sub>ng sca</sub>li<sub>ng</sub> t<sub>o</sub> l<sub>arge num</sub>b<sub>ers o</sub>f d<sub>ev</sub>i<sub>ces.</sub> T<sub>o a</sub>dd<sub>ress</sub> th<sub>e</sub> l<sub>a-</sub> tency bottleneck, Thapa et al. [39] proposed SFL, which combines SL-st<sub>y</sub>le model <sub>p</sub>artitionin<sub>g</sub> with FL-st<sub>y</sub>le <sub>p</sub>arallel client trainin<sub>g</sub>. B<sub>y</sub> allowin<sub>g</sub> multi<sub>p</sub>le clients to <sub>p</sub>erform forward <sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ation in <sub>p</sub>arallel and a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> their <sub>g</sub>radients<sub>,</sub> SFL si<sub>g</sub>nificantl<sub>y</sub> reduces the trainin<sub>g</sub> time com<sub>p</sub>ared to vanilla SL. Buildin<sub>g</sub> u<sub>p</sub>on the <sub>p</sub>arall<sub>e</sub>li<sub>za</sub>ti<sub>on</sub> <sub>concep</sub>t<sub>s</sub> <sub>o</sub>f SFL<sub>,</sub> <sub>recen</sub>t <sub>researc</sub>h h<sub>as</sub> f<sub>ocuse</sub>d <sub>on</sub> f<sub>ur</sub>th<sub>er</sub> streamlining the training pipeline. Joshi et al. [15] and Kimet al. [17] intr<sub>o</sub>d<sub>uce</sub>d th<sub>e</sub> PSL fr<sub>a</sub>m<sub>ewo</sub>rk<sub>.</sub> Unlik<sub>e</sub> SFL<sub>,</sub> PSL r<sub>e</sub>m<sub>oves</sub> th<sub>e</sub> r<sub>equ</sub>ir<sub>e</sub>- ment for client-side model a<sub>gg</sub>re<sub>g</sub>ation. This architectural chan<sub>g</sub>e <sub>s</sub>i<sub>mp</sub>lifi<sub>es</sub> th<sub>e sync</sub>h<sub>ron</sub>i<sub>za</sub>ti<sub>on process an</sub>d <sub>acce</sub>l<sub>era</sub>t<sub>es mo</sub>d<sub>e</sub>l <sub>con-</sub> <sub>vergence.</sub> R<sub>ecen</sub>t <sub>wor</sub>k h<sub>ave</sub> f<sub>ur</sub>th<sub>er</sub> i<sub>mprove</sub>d th<sub>e compu</sub>t<sub>a</sub>ti<sub>ona</sub>l and communication eficienc<sub>y</sub> of SL b<sub>y</sub> o<sub>p</sub>timizin<sub>g</sub> client-side <sub>g</sub>radient a<sub>gg</sub>re<sub>g</sub>ation<sub>,</sub> desi<sub>g</sub>nin<sub>g</sub> scalable server-side u<sub>p</sub>date mechanisms<sub>,</sub> and ado<sub>p</sub>tin<sub>g</sub> ada<sub>p</sub>tive model <sub>p</sub>artitionin<sub>g</sub>.

## 2.2 Data Heterogeneity Mitigation

Research on miti<sub>g</sub>atin<sub>g</sub> data hetero<sub>g</sub>eneit<sub>y</sub> in distributed learnin<sub>g</sub> has been lar<sub>g</sub>el<sub>y</sub> driven b<sub>y</sub> FL<sub>, p</sub>rimaril<sub>y</sub> throu<sub>g</sub>h a<sub>gg</sub>re<sub>g</sub>ati<sub>o</sub>n <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd <sub>pe</sub>r<sub>so</sub>n<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>ve</sub> m<sub>e</sub>th<sub>o</sub>d<sub>s</sub> such as SCAFFOLD [16], FedD<sub>y</sub>n [1], and FedDC [11] alleviate client drift via control variates or re<sub>g</sub>ularization<sub>,</sub> while <sub>p</sub>ersonalization a<sub>pp</sub>roaches (e.<sub>g</sub>., FedAPA [37], FedCPD [14]) enhance ada<sub>p</sub>tabilit<sub>y</sub> under non-IID data throu<sub>g</sub>h ada<sub>p</sub>tive a<sub>gg</sub>re<sub>g</sub>ation or <sub>pro</sub>t<sub>o</sub>t<sub>ype-</sub>b<sub>ase</sub>d l<sub>earn</sub>i<sub>ng.</sub> M<sub>ore</sub> <sub>recen</sub>tl<sub>y,</sub> <sub>severa</sub>l <sub>s</sub>t<sub>u</sub>di<sub>es</sub> h<sub>ave</sub> <sub>a</sub>d<sub>-</sub> d<sub>resse</sub>d d<sub>a</sub>t<sub>a</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> th<sub>roug</sub>h <sub>up</sub>d<sub>a</sub>t<sub>e cons</sub>i<sub>s</sub>t<sub>ency an</sub>d <sub>con</sub>fli<sub>c</sub>t<sub>-</sub> aware o<sub>p</sub>timization. Methods such as FedProx [24], FedNova [42], and MOON [23] im<sub>p</sub>rove trainin<sub>g</sub> stabilit<sub>y</sub> b<sub>y</sub> constrainin<sub>g</sub> local <sup>d</sup>ri<sup>f</sup>t, correcting o<sup>b</sup>jective inconsistency, or en<sup>h</sup>ancing representation consistenc<sub>y</sub>. Conflict-aware a<sub>pp</sub>roaches, includin<sub>g</sub> PCGrad [55],

![](images/a077d5daa5d67abeadc9de8969db3f969c5b6fa4b213d2c8f99f1d134b35f446.jpg)

![](images/f2514654cfd4358dcd1ac5912d8a41422c063d1c283ace2efdae7f966f16f716.jpg)

Figure 1: Performance of PSL and SFL on CIFAR-10 datasets under non-IID settings with ResNet-18.  
![](images/c595b3ed6eca45855a545be1d2a790226a9ddc47354300e03622b80089741a9a.jpg)  
(a) Test accuracy

![](images/e4ab4c5f72a8bcaaeaeb8ea407efa99611aa4f1f216e1a1cee8e8e40aa70ab53.jpg)  
(b) Conflict ratio  
Figure 2: Gradient conflict ratio and accuracy of PSL and SFL under diferent non-IID levels on CIFAR-10 with ResNet-18.

FedIIR [12], FedLD [57], and FedCALM [58], further miti<sub>g</sub>ate interference amon<sub>g</sub> client u<sub>p</sub>dates b<sub>y</sub> ali<sub>g</sub>nin<sub>g</sub> o<sub>p</sub>timization directions or su<sub>pp</sub>ressin<sub>g</sub> conflictin<sub>g</sub> com<sub>p</sub>onents. However<sub>,</sub> these methods <sub>are</sub> <sub>ma</sub>i<sub>n</sub>l<sub>y</sub> d<sub>es</sub>i<sub>gne</sub>d f<sub>or</sub> FL<sub>,</sub> <sub>w</sub>h<sub>ere</sub> f<sub>u</sub>ll<sub>-mo</sub>d<sub>e</sub>l <sub>rep</sub>li<sub>ca</sub>ti<sub>on</sub> <sub>an</sub>d it<sub>-</sub> erative client a<sub>gg</sub>re<sub>g</sub>ation are assumed. B<sub>y</sub> contrast<sub>,</sub> PSL ado<sub>p</sub>ts a <sub>p</sub>artitioned architecture without client-side a<sub>gg</sub>re<sub>g</sub>ation<sub>,</sub> leadin<sub>g</sub> to distinct <sub>g</sub>radient interaction <sub>p</sub>atterns. Therefore<sub>,</sub> miti<sub>g</sub>atin<sub>g</sub> <sub>g</sub>radi<sub>e</sub>nt in<sub>co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>e</sub>n<sub>cy</sub> in PSL r<sub>equ</sub>ir<sub>es</sub> m<sub>e</sub>th<sub>o</sub>d<sub>s</sub> t<sub>a</sub>il<sub>o</sub>r<sub>e</sub>d t<sub>o</sub> it<sub>s</sub> <sub>u</sub>ni<sub>que</sub> trainin<sub>g</sub> d<sub>y</sub>namics.

## 3 Background and Motivation

PSL has emer<sub>g</sub>ed as a <sub>p</sub>romisin<sub>g</sub> trainin<sub>g</sub> <sub>p</sub>aradi<sub>g</sub>m for resourceconstrained ed<sub>g</sub>e intelli<sub>g</sub>ence. B<sub>y</sub> <sub>p</sub>artitionin<sub>g</sub> the model between e<sup>d</sup>ge <sup>d</sup>evices an<sup>d</sup> t<sup>h</sup>e server, PSL o<sup>fl</sup>oa<sup>d</sup>s t<sup>h</sup>e majority o<sup>f</sup> computati<sub>on</sub> <sub>an</sub>d <sub>memory</sub> <sub>over</sub>h<sub>ea</sub>d f<sub>rom</sub> <sub>c</sub>li<sub>en</sub>t<sub>s.</sub> C<sub>ruc</sub>i<sub>a</sub>ll<sub>y,</sub> b<sub>y</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>l<sub>y</sub> eliminatin<sub>g</sub> client-side <sub>p</sub>arameter a<sub>gg</sub>re<sub>g</sub>ation<sub>,</sub> PSL sim<sub>p</sub>lifies the tr<sub>a</sub>inin<sub>g</sub> <sub>p</sub>i<sub>pe</sub>lin<sub>e</sub> <sub>a</sub>nd minimiz<sub>es</sub> <sub>co</sub>mm<sub>u</sub>ni<sub>ca</sub>ti<sub>o</sub>n <sub>ove</sub>rh<sub>ea</sub>d<sub>,</sub> m<sub>a</sub>kin<sub>g</sub> it hi<sub>g</sub>hl<sub>y</sub> <sub>a</sub>tt<sub>rac</sub>ti<sub>ve</sub> f<sub>or</sub> l<sub>arge-sca</sub>l<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d d<sub>ep</sub>l<sub>oymen</sub>t <sub>compare</sub>d to conventional SFL. However<sub>,</sub> this hi<sub>g</sub>hl<sub>y</sub> eficient<sub>,</sub> client-side <sub>p</sub>arameter a<sub>gg</sub>re<sub>g</sub>ation-free architecture introduces an inherent o<sub>p</sub>ti mization <sub>vu</sub>lnerabilit<sub>y u</sub>nder statistical hetero<sub>g</sub>eneit<sub>y</sub>. In PSL<sub>,</sub> each <sub>c</sub>li<sub>en</sub>t <sub>execu</sub>t<sub>es</sub> f<sub>orwar</sub>d <sub>an</sub>d b<sub>ac</sub>k<sub>war</sub>d <sub>propaga</sub>ti<sub>on</sub> b<sub>ase</sub>d <sub>so</sub>l<sub>e</sub>l<sub>y on</sub> its local<sub>,</sub> t<sub>yp</sub>icall<sub>y</sub> non-IID data<sub>,</sub> u<sub>p</sub>datin<sub>g</sub> its sub-model without <sub>any cross-c</sub>li<sub>en</sub>t <sub>sync</sub>h<sub>ron</sub>i<sub>za</sub>ti<sub>on.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> l<sub>oca</sub>l <sub>su</sub>b<sub>-mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>rap</sub>idl<sub>y over</sub>fit t<sub>o</sub> l<sub>oca</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons.</sub> Thi<sub>s</sub> l<sub>oca</sub>l bi<sub>as causes c</sub>li<sub>en</sub>t<sub>s</sub> to up<sup>l</sup>oa<sup>d</sup> semantica<sup>ll</sup>y <sup>d</sup>isjointe<sup>d</sup> interme<sup>d</sup>iate activations to t<sup>h</sup>e <sub>se</sub>r<sub>ve</sub>r<sub>.</sub> Und<sub>e</sub>r n<sub>o</sub>n-IID <sub>se</sub>ttin<sub>gs,</sub> thi<sub>s se</sub>m<sub>a</sub>nti<sub>c</sub> mi<sub>s</sub>m<sub>a</sub>t<sub>c</sub>h tr<sub>a</sub>n<sub>s</sub>l<sub>a</sub>t<sub>es</sub> into hi<sub>g</sub>hl<sub>y</sub> diver<sub>g</sub>ent <sub>g</sub>radients durin<sub>g</sub> server-side back<sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ati<sub>on.</sub> I<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>a g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y ca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on s</sub>i<sub>gna</sub>l<sub>,</sub> th<sub>e server</sub> receives conflictin<sub>g</sub> u<sub>p</sub>dates that accumulate over time<sub>,</sub> leadin<sub>g</sub> to o<sub>p</sub>timization diver<sub>g</sub>ence and severe client drift.

This vulnerabilit<sub>y</sub> is su<sub>pp</sub>orted b<sub>y</sub> our <sub>p</sub>reliminar<sub>y</sub> ex<sub>p</sub>eriments. A<sub>s</sub> <sub>s</sub>h<sub>ow</sub>n in Fi<sub>g.</sub> 1<sub>,</sub> <sub>o</sub>n n<sub>o</sub>n-IID CIFAR-10 <sub>w</sub>ith R<sub>es</sub>N<sub>e</sub>t-18<sub>,</sub> PSL <sub>co</sub>nver<sub>g</sub>es more slowl<sub>y</sub> and less stabl<sub>y</sub> than SFL<sub>,</sub> su<sub>gg</sub>estin<sub>g</sub> that the absence of client-side a<sub>gg</sub>re<sub>g</sub>ation increases its sensitivit<sub>y</sub> to data hetero<sub>g</sub>eneit<sub>y</sub>. Fi<sub>g</sub>. 2 further shows that stron<sub>g</sub>er hetero<sub>g</sub>eneit<sub>y</sub> l<sub>ea</sub>d<sub>s</sub> t<sub>o a</sub> hi<sub>g</sub>h<sub>er gra</sub>di<sub>en</sub>t <sub>con</sub>fli<sub>c</sub>t <sub>ra</sub>ti<sub>o an</sub>d l<sub>ower</sub> t<sub>es</sub>t <sub>accuracy,</sub> <sub>w</sub>ith <sub>a</sub> m<sub>o</sub>r<sub>e</sub> <sub>p</sub>r<sub>o</sub>n<sub>ou</sub>n<sub>ce</sub>d d<sub>eg</sub>r<sub>a</sub>d<sub>a</sub>ti<sub>o</sub>n in PSL<sub>.</sub> Th<sub>ese</sub> r<sub>esu</sub>lt<sub>s</sub> indi<sub>ca</sub>t<sub>e</sub> th<sub>a</sub>t PSL’<sub>s</sub> <sub>per</sub>f<sub>ormance</sub> l<sub>oss</sub> i<sub>s</sub> <sub>c</sub>l<sub>ose</sub>l<sub>y</sub> <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>cross-c</sub>li<sub>en</sub>t

![](images/485e034d8a7f3afbecdca1d3db42f5220478ed62e221d12a93c280e26092d712.jpg)  
Figure 3: An overview of PSL-RGHA architecture.

![](images/a2d63a7e4f84eacafcd50f136a0f70ac35e4110340b7631c73f66b84d82a24a7.jpg)  
(a) Individual performance

![](images/3c6550f7bcfe745328712715fc601cc22d33e59cd67fb54306b636bc750aca53.jpg)  
(b) Accuracy vs. Epochs  
50) <sup>Device</sup> <sup>1</sup>Figure 4: Performance of PSL using single client device gradiy <sup>( Device</sup> <sup>3</sup>ent updates on CIFAR-10 under non-IID settings (ResNet-18).

A<sub>g</sub>radient inconsistenc<sub>y,</sub> which introduces nois<sub>y</sub> and unstable server-20 30 40 50si<sup>d</sup>e up<sup>d</sup>ates. T<sup>h</sup>e a<sup>b</sup>ove o<sup>b</sup>servations revea<sup>l</sup> a major <sup>l</sup>imitation of existin<sub>g</sub> PSL frameworks: eliminatin<sub>g</sub> client-side a<sub>gg</sub>re<sub>g</sub>ation maximizes communication eficienc<sub>y</sub> but simultaneousl<sub>y</sub> removes th<sub>e essen</sub>ti<sub>a</sub>l <sub>cross-c</sub>li<sub>en</sub>t <sub>ca</sub>lib<sub>ra</sub>ti<sub>on nee</sub>d<sub>e</sub>d f<sub>or s</sub>t<sub>a</sub>bl<sub>e convergence.</sub> Conse<sub>q</sub>uentl<sub>y,</sub> hetero<sub>g</sub>eneous client u<sub>p</sub>dates can interfere with each other<sub>,</sub> causin<sub>g</sub> oscillation<sub>,</sub> slow conver<sub>g</sub>ence<sub>,</sub> or even conver<sub>g</sub>ence f<sub>a</sub>il<sub>ure.</sub> Th<sub>ere</sub>f<sub>ore,</sub> th<sub>e</sub> k<sub>ey c</sub>h<sub>a</sub>ll<sub>enge</sub> i<sub>s</sub> h<sub>ow</sub> t<sub>o preserve</sub> th<sub>e e</sub>f<sub>-</sub> ficienc<sub>y</sub> advanta<sub>g</sub>e of a<sub>gg</sub>re<sub>g</sub>ation-free PSL while still enforcin<sub>g</sub> directional consistenc<sub>y</sub> amon<sub>g</sub> hetero<sub>g</sub>eneous client <sub>g</sub>radients. This <sub>compe</sub>lli<sub>ng</sub> dil<sub>emma</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>es</sub> th<sub>e</sub> <sub>necess</sub>it<sub>y</sub> <sub>o</sub>f <sub>our</sub> <sub>researc</sub>h<sub>:</sub> d<sub>e-</sub> si<sub>g</sub>nin<sub>g</sub> a robust server-side coordination mechanism ca<sub>p</sub>able of i<sub>n</sub>f<sub>err</sub>i<sub>ng</sub> <sub>a</sub> <sub>re</sub>li<sub>a</sub>bl<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l d<sub>escen</sub>t t<sub>ren</sub>d f<sub>rom</sub> <sub>no</sub>i<sub>sy</sub> <sub>c</sub>li<sub>en</sub>t <sub>s</sub>i<sub>gna</sub>l<sub>s,</sub> and activel<sub>y</sub> su<sub>pp</sub>ressin<sub>g</sub> conflictin<sub>g</sub> u<sub>p</sub>date directions to <sub>g</sub>uarantee stable<sub>,</sub> a<sub>gg</sub>re<sub>g</sub>ation-free distributed trainin<sub>g</sub>.

## 4 System Design

## 4.1 Overview

In thi<sub>s sec</sub>ti<sub>o</sub>n<sub>, we p</sub>r<sub>ese</sub>nt PSL-RGHA<sub>, a</sub> r<sub>o</sub>b<sub>us</sub>t <sub>g</sub>r<sub>a</sub>di<sub>e</sub>nt-<sub>a</sub>li<sub>g</sub>nm<sub>e</sub>nt framework en<sub>g</sub>ineered to stabilize PSL under severe statistical het-<sub>e</sub>r<sub>oge</sub>n<sub>e</sub>it<sub>y.</sub> A<sub>s</sub> ill<sub>us</sub>tr<sub>a</sub>t<sub>e</sub>d in Fi<sub>g.</sub> 3<sub>,</sub> PSL-RGHA int<sub>eg</sub>r<sub>a</sub>t<sub>es</sub> t<sub>wo</sub> <sub>sy</sub>ner<sub>g</sub>istic modules within an a<sub>gg</sub>re<sub>g</sub>ation-free architecture: i) RGPC (Section 4.2), which s<sub>y</sub>nthesizes a <sub>g</sub>lobal reference direction b<sub>y p</sub>ro-<sub>g</sub>ressivel<sub>y</sub> a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> consistent client <sub>g</sub>radients via Weiszfeld u<sub>p</sub>dates; ii)GHA (Section 4.3), which levera<sub>g</sub>es the reference <sub>g</sub>radi-<sub>en</sub>t t<sub>o en</sub>f<sub>orce g</sub>l<sub>o</sub>b<sub>a</sub>l di<sub>rec</sub>ti<sub>ona</sub>l <sub>cons</sub>i<sub>s</sub>t<sub>ency.</sub>

In PSL-RGHA<sub>,</sub> th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l i<sub>s sp</sub>lit int<sub>o c</sub>li<sub>e</sub>nt-<sub>s</sub>id<sub>e su</sub>b-m<sub>o</sub>d<sub>e</sub>l<sub>s</sub> <sub>an</sub>d <sub>a s</sub>h<sub>are</sub>d <sub>server-s</sub>id<sub>e su</sub>b<sub>-mo</sub>d<sub>e</sub>l<sub>.</sub> I<sub>n eac</sub>h t<sub>ra</sub>i<sub>n</sub>i<sub>ng roun</sub>d<sub>, c</sub>li<sub>en</sub>t<sub>s</sub> <sub>per</sub>f<sub>orm</sub> f<sub>orwar</sub>d <sub>propaga</sub>ti<sub>on</sub> <sub>on</sub> l<sub>oca</sub>l d<sub>a</sub>t<sub>a</sub> <sub>an</sub>d <sub>up</sub>l<sub>oa</sub>d <sub>cu</sub>t<sub>-</sub>l<sub>ayer</sub> <sub>ac</sub>ti<sub>va</sub>ti<sub>ons</sub> t<sub>o</sub> th<sub>e</sub> <sub>server.</sub> Th<sub>e</sub> <sub>server</sub> th<sub>en</sub> <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> f<sub>orwar</sub>d <sub>an</sub>d backward <sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ation to obtain client-s<sub>p</sub>ecific server <sub>g</sub>radients. RGPC <sub>co</sub>n<sub>s</sub>tr<sub>uc</sub>t<sub>s a</sub> r<sub>o</sub>b<sub>us</sub>t r<sub>e</sub>f<sub>e</sub>r<sub>e</sub>n<sub>ce g</sub>r<sub>a</sub>di<sub>e</sub>nt<sub>, a</sub>nd GHA <sub>a</sub>li<sub>g</sub>n<sub>s eac</sub>h <sub>c</sub>li<sub>en</sub>t <sub>gra</sub>di<sub>en</sub>t <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>re</sub>f<sub>erence</sub> h<sub>a</sub>lf<sub>-space.</sub> Th<sub>e</sub> <sub>server up</sub>d<sub>a</sub>t<sub>es</sub> th<sub>e s</sub>h<sub>are</sub>d <sub>su</sub>b<sub>-mo</sub>d<sub>e</sub>l <sub>us</sub>i<sub>ng</sub> th<sub>e correc</sub>t<sub>e</sub>d <sub>gra</sub>di<sub>en</sub>t<sub>s</sub> <sub>an</sub>d <sub>re</sub>t<sub>urns</sub> th<sub>e</sub> <sub>cu</sub>t<sub>-</sub>l<sub>ayer</sub> <sub>gra</sub>di<sub>en</sub>t<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>c</sub>li<sub>en</sub>t<sub>s,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> b<sub>ac</sub>k<sub>propaga</sub>ti<sub>on</sub> <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r</sub> l<sub>oca</sub>l <sub>su</sub>b<sub>-mo</sub>d<sub>e</sub>l<sub>s.</sub>

![](images/b4ed94083bd05b5f426c4e781705403d0629ec7ed41f106bf2a11f7a56e69d92.jpg)  
Figure 5: An illustration of RGPC: Gradients are scored by directional consistency (left) and aggregated via ranked Weiszfeld updates (right) to construct the reference gradient.

## 4.2 Reference Gradient Progressive Construction

In PSL<sub>,</sub> the absence of client-side a<sub>gg</sub>re<sub>g</sub>ation severel<sub>y</sub> exacerbates inter-client <sub>g</sub>radient misali<sub>g</sub>nment<sub>,</sub> <sub>p</sub>roducin<sub>g</sub> conflictin<sub>g</sub> u<sub>p</sub>dates <sub>a</sub>t th<sub>e</sub> <sub>server.</sub> T<sub>o</sub> <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>va</sub>lid<sub>a</sub>t<sub>e</sub> thi<sub>s,</sub> <sub>we</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d th<sub>e</sub> <sub>op</sub>ti mization trajectories o<sup>f</sup> in<sup>d</sup>ivi<sup>d</sup>ua<sup>l</sup> c<sup>l</sup>ients starting <sup>f</sup>rom i<sup>d</sup>entica<sup>l</sup> initi<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub> A<sub>s</sub> ill<sub>us</sub>tr<sub>a</sub>t<sub>e</sub>d in Fi<sub>g.</sub> 4<sub>a,</sub> th<sub>e</sub> r<sub>esu</sub>lt<sub>a</sub>nt t<sub>es</sub>t <sub>accu</sub>r<sub>a</sub>- cies exhibit extreme s<sub>p</sub>atial variance<sub>,</sub> confirmin<sub>g</sub> the severe hetero<sub>g</sub>eneit<sub>y</sub> of local <sub>g</sub>radients. Furthermore<sub>,</sub> the client contributin<sub>g</sub> the maximum <sub>p</sub>erformance <sub>g</sub>ain fluctuates across e<sub>p</sub>ochs (Fi<sub>g</sub>. 4b), indicatin<sub>g</sub> that no sin<sub>g</sub>le client consistentl<sub>y</sub> ali<sub>g</sub>ns with the o<sub>p</sub>timal <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l d<sub>escen</sub>t di<sub>rec</sub>ti<sub>on.</sub> C<sub>onsequen</sub>tl<sub>y, un</sub>d<sub>er suc</sub>h di<sub>rec</sub>ti<sub>ona</sub>l <sub>con</sub> flict<sub>,</sub> naive uniform a<sub>gg</sub>re<sub>g</sub>ation inadvertentl<sub>y</sub> incor<sub>p</sub>orates these misali<sub>g</sub>ned <sub>g</sub>radients as detrimental noise<sub>,</sub> thereb<sub>y</sub> de<sub>g</sub>radin<sub>g</sub> the <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>an</sub>d <sub>s</sub>t<sub>a</sub>lli<sub>ng</sub> <sub>convergence.</sub>

To this end<sub>,</sub> we <sub>p</sub>ro<sub>p</sub>ose RGPC<sub>,</sub> a robust a<sub>gg</sub>re<sub>g</sub>ation mechanism that ranks client <sub>g</sub>radients via a directional consistenc<sub>y</sub> score and <sub>p</sub>ro<sub>g</sub>ressivel<sub>y</sub> s<sub>y</sub>nthesizes them usin<sub>g</sub> Weiszfeld u<sub>p</sub>dates [46] , as illustrated in Fi<sub>g</sub>. 5. This <sub>p</sub>rocedure <sub>y</sub>ields a robust reference <sub>g</sub>radi<sub>en</sub>t th<sub>a</sub>t <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> filt<sub>ers ou</sub>t <sub>no</sub>i<sub>se an</sub>d <sub>cap</sub>t<sub>ures</sub> th<sub>e un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> l<sub>o</sub>b<sub>a</sub>l <sub>conver ence</sub> t<sub>ren</sub>d<sub>, w</sub>hi<sub>c</sub>h i<sub>s e</sub>l<sub>a</sub>b<sub>ora</sub>t<sub>e</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows.</sub>

4.2.1 Gradient Consistency Scoring. Existing methods often estimate client contributions usin<sub>g</sub> <sub>g</sub>radient ma<sub>g</sub>nitudes<sub>,</sub> assumin<sub>g</sub> that lar<sub>g</sub>er u<sub>p</sub>dates are more informative [3, 41, 53]. However, un der statistical hetero<sub>g</sub>eneit<sub>y</sub> (non-IID data), <sub>g</sub>radients with lar<sub>g</sub>e <sub>magn</sub>it<sub>u</sub>d<sub>es o</sub>ft<sub>en s</sub>t<sub>em</sub> f<sub>rom</sub> l<sub>oca</sub>l <sub>over</sub>fitti<sub>ng or ou</sub>tli<sub>ers, w</sub>hi<sub>c</sub>h are directionall<sub>y</sub> inconsistent with the <sub>g</sub>lobal o<sub>p</sub>timization landsca<sub>p</sub>e and ma<sub>y</sub> im<sub>p</sub>ede conver<sub>g</sub>ence. Conversel<sub>y,</sub> <sub>g</sub>radients with <sub>sma</sub>ll<sub>er</sub> <sub>magn</sub>it<sub>u</sub>d<sub>es</sub> b<sub>u</sub>t <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t di<sub>rec</sub>ti<sub>ons</sub> <sub>o</sub>ft<sub>en</sub> <sub>prov</sub>id<sub>e</sub> <sub>re</sub>li<sub>-</sub> able <sub>p</sub>ro<sub>g</sub>ress. This necessitates a metric that <sub>p</sub>rioritizes directional ali<sub>g</sub>nment over scalar ma<sub>g</sub>nitude. Accordin<sub>g</sub>l<sub>y,</sub> we introduce the gradient consistency score, a metric designed to quantify the cross-<sub>c</sub>li<sub>en</sub>t di<sub>rec</sub>ti<sub>ona</sub>l <sub>agreemen</sub>t<sub>.</sub> F<sub>orma</sub>ll<sub>y,</sub> f<sub>or</sub> th<sub>e</sub> �<sub>-</sub>th t<sub>ra</sub>i<sub>n</sub>i<sub>ng roun</sub>d<sub>,</sub> the consistenc<sub>y</sub> score $S _ { i } ^ { t }$ <sub>o</sub>f th<sub>e</sub> �<sub>-</sub>th <sub>c</sub>li<sub>en</sub>t i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> th<sub>e</sub> <sub>average</sub> <sub>cos</sub>i<sub>ne s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> b<sub>e</sub>t<sub>ween</sub> it<sub>s</sub> l<sub>oca</sub>l <sub>gra</sub>di<sub>en</sub>t <sub>an</sub>d th<sub>ose o</sub>f <sub>a</sub>ll <sub>o</sub>th<sub>er</sub> <sub>p</sub>artici<sub>p</sub>atin<sub>g</sub> clients:

$$
S _ { i } ^ { t } = \frac { 1 } { \left| \mathcal { M } \right| - 1 } \sum _ { j \neq i } \frac { \left. g _ { i } ^ { t } , g _ { j } ^ { t } \right. } { \left\| g _ { i } ^ { t } \right\| \cdot \left\| g _ { j } ^ { t } \right\| } .\tag{1}
$$

![](images/c845e489a5c7c471c5e6997b2e0e86013637f0f183ae7433444a0a5b182e65f7.jpg)

![](images/fadb4f1d262a9e84994c1fd8e0d8d0cd756481483bf96449de8abe6e39f3ecc4.jpg)  
(a) Consistency scores  
(b) Accuracy  
Figure 6: The gradient consistency scores and test accuracies of client devices on CIFAR-10 (ResNet-18).

<sub>c</sub><sup>u</sup><sub>w</sub>h<sub>ere</sub> $M = \{ 1 , 2 , \dots , M \}$ denotes the set of <sub>p</sub>artici<sub>p</sub>atin<sub>g</sub> client d<sub>ev</sub>i<sub>ces,</sub> $\langle \cdot , \cdot \rangle$ denotes the inner <sub>p</sub>roduct, ∥·∥ is the Euclidean norm, <sub>an</sub>d $g _ { k } ^ { t }$ <sub>Device</sub> <sub>1 Device</sub> <sub>2 Device</sub> <sub>3</sub>i<sub>s</sub> th<sub>e gra</sub>di<sub>en</sub>t <sub>o</sub>f th<sub>e</sub> �<sub>-</sub>th <sub>c</sub>li<sub>en</sub>t d<sub>ev</sub>i<sub>ce a</sub>t th<sub>e</sub> �<sub>-</sub>th t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>roun</sub>d<sub>.</sub>

E<sub>q</sub>u. (1) <sub>p</sub>rovides a <sub>q</sub>uantitative measure of directional centralit<sub>y</sub>. A hi<sub>g</sub>h<sub>e</sub>r $S _ { i } ^ { t }$ i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> th<sub>a</sub>t th<sub>e gra</sub>di<sub>en</sub>t <sub>o</sub>f <sub>c</sub>li<sub>en</sub>t � <sub>a</sub>li<sub>gns c</sub>l<sub>ose</sub>l<sub>y w</sub>ith the <sub>p</sub>revailin<sub>g</sub> descent trend of the <sub>p</sub>o<sub>p</sub>ulation<sub>,</sub> su<sub>gg</sub>estin<sub>g</sub> a con-<sub>s</sub>t<sub>ruc</sub>ti<sub>ve con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> C<sub>onverse</sub>l<sub>y, a</sub> l<sub>ower</sub> $S _ { i } ^ { t }$ si<sub>g</sub>nifies weak directional a<sub>g</sub>reement<sub>,</sub> im<sub>p</sub>l<sub>y</sub>in<sub>g</sub> that the u<sub>p</sub>date ma<sub>y</sub> introduce noise or conflictin<sub>g</sub> si<sub>g</sub>nals that decelerate conver<sub>g</sub>ence. A<sub>s</sub> <sub>s</sub>h<sub>ow</sub>n in Fi<sub>g.</sub> $^ { 6 , }$ clients with hi<sub>g</sub>her consistenc<sub>y</sub> scores consistentl<sub>y</sub> <sub>y</sub>ield hi<sub>g</sub>her test accurac<sub>y</sub> in sin<sub>g</sub>le-client u<sub>p</sub>date studies<sub>,</sub> validatin<sub>g</sub> $S _ { i } ^ { t }$ <sub>as a</sub>n <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> indi<sub>ca</sub>t<sub>o</sub>r <sub>o</sub>f <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n <sub>u</sub>tilit<sub>y.</sub>

After com<sub>p</sub>utin<sub>g</sub> $\{ S _ { i } ^ { t } \} _ { i \in { M } } ,$ <sub>,</sub> th<sub>e server ran</sub>k<sub>s c</sub>li<sub>en</sub>t<sub>s</sub> i<sub>n</sub> d<sub>escen</sub>di<sub>ng</sub> <sub>or</sub>d<sub>er an</sub>d <sub>o</sub>bt<sub>a</sub>i<sub>ns</sub> i<sub>n</sub>di<sub>ces</sub> $( r _ { 1 } , r _ { 2 } \ldots , r _ { M } )$ <sub>suc</sub>h th<sub>a</sub>t

$$
S _ { r _ { 1 } } ^ { t } \geq S _ { r _ { 2 } } ^ { t } \geq \cdots \geq S _ { r _ { M } } ^ { t } ,\tag{2}
$$

<sub>w</sub>h<sub>ere</sub> $r _ { 1 }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>c</sub>li<sub>en</sub>t <sub>w</sub>ith th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t di<sub>rec</sub>ti<sub>ona</sub>l <sub>cons</sub>i<sub>s</sub>t<sub>ency</sub> at round �. This rankin<sub>g</sub> serves as the basis for RGPC<sub>,</sub> allowin<sub>g</sub> the s<sub>y</sub>stem to <sub>p</sub>rioritize directionall<sub>y</sub> reliable <sub>g</sub>radients while miti<sub>g</sub>atin<sub>g</sub> th<sub>e</sub> i<sub>mpac</sub>t <sub>o</sub>f <sub>ou</sub>tli<sub>ers</sub> d<sub>ur</sub>i<sub>ng</sub> th<sub>e</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>re</sub>f<sub>erence</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on.</sub>

4.2.2 Progressive Construction via WeiszfeldUpdates. To synthesize <sub>a ro</sub>b<sub>us</sub>t <sub>re</sub>f<sub>erence gra</sub>di<sub>en</sub>t f<sub>rom</sub> th<sub>e</sub> h<sub>e</sub>t<sub>erogeneous c</sub>li<sub>en</sub>t <sub>up</sub>d<sub>a</sub>t<sub>es,</sub> we em<sub>p</sub>lo<sub>y</sub> a <sub>p</sub>ro<sub>g</sub>ressive Weiszfeld [47] scheme that <sub>p</sub>rioritizes directionall<sub>y</sub> consistent <sub>g</sub>radients (i.e., those with hi<sub>g</sub>her consis-<sup>tenc</sup>y <sup>scores</sup> $S _ { i } ^ { t } )$ . Unlike sim<sub>p</sub>le a<sub>v</sub>era<sub>g</sub>in<sub>g</sub> <sub>w</sub>hich is sensiti<sub>v</sub>e to outliers<sub>,</sub> this a<sub>pp</sub>roach se<sub>q</sub>uentiall<sub>y</sub> a<sub>gg</sub>re<sub>g</sub>ates ranked directions to a<sub>pp</sub>roximate the <sub>g</sub>eometric median<sub>,</sub> which is robust es<sub>p</sub>eciall<sub>y</sub> in hi<sub>g</sub>h-dimensional s<sub>p</sub>aces.

Given the set of client <sub>g</sub>radients $\{ g _ { i } ^ { t } \} _ { i \in \mathcal { M } }$ <sub>a</sub>t <sub>roun</sub>d �<sub>, we</sub> fi<sub>rs</sub>t decou<sub>p</sub>le directional information from ma<sub>g</sub>nitude b<sub>y</sub> normalizin<sub>g</sub> <sub>va</sub>lid <sub>gra</sub>di<sub>en</sub>t<sub>s</sub> <sub>on</sub>t<sub>o</sub> th<sub>e</sub> <sub>un</sub>it <sub>sp</sub>h<sub>ere:</sub>

$$
u _ { i } ^ { t } = \frac { g _ { i } ^ { t } } { \lVert g _ { i } ^ { t } \rVert } , \quad \forall i \in \mathcal { M } , \mathrm { ~ s . t . ~ } \lVert g _ { i } ^ { t } \rVert > \delta ,\tag{3}
$$

<sub>w</sub>h<sub>ere</sub> � i<sub>s a</sub> th<sub>res</sub>h<sub>o</sub>ld t<sub>o</sub> filt<sub>er numer</sub>i<sub>ca</sub>ll<sub>y uns</sub>t<sub>a</sub>bl<sub>e up</sub>d<sub>a</sub>t<sub>es.</sub>

Let $( r _ { 1 } , r _ { 2 } \ldots , r _ { M } )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> i<sub>n</sub>di<sub>ces o</sub>f <sub>c</sub>li<sub>en</sub>t<sub>s sor</sub>t<sub>e</sub>d b th<sub>e</sub>i<sub>r</sub> consistenc<sub>y</sub> scores in descendin<sub>g</sub> order. We <sub>p</sub>rocess clients accordin<sub>g</sub> t<sub>o</sub> thi<sub>s</sub> r<sub>a</sub>nk<sub>,</sub> m<sub>a</sub>int<sub>a</sub>inin<sub>g a</sub>n <sub>evo</sub>l<sub>v</sub>in<sub>g es</sub>tim<sub>a</sub>t<sub>e</sub> $\pmb { g } \in \mathbb { R } ^ { d }$ con-<sub>s</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>w</sub>ithi<sub>n</sub> th<sub>e convex un</sub>it b<sub>a</sub>ll $C \ \triangleq \ \left\{ \pmb { x } \in \mathbb { R } ^ { d } : \| \pmb { x } \| \leq 1 \right\}$ Cruciall<sub>y</sub>, we <sub>p</sub>erform intermediate u<sub>p</sub>dates within C (a convex set) rather than the unit s<sub>p</sub>here (non-convex) to <sub>g</sub>uarantee stable conver<sub>g</sub>ence <sub>p</sub>ro<sub>p</sub>erties. For the cumulative subset $\mathcal { P } _ { k } ^ { t } = \{ u _ { r _ { 1 } } ^ { t } , \ldots , u _ { r _ { k } } ^ { t } \}$ o<sup>f</sup> t<sup>h</sup>e top-<sup>�</sup> c<sup>l</sup>ients, t<sup>h</sup>e smoot<sup>h</sup>e<sup>d</sup> geometric-me<sup>d</sup>ian o<sup>b</sup>jective over C is defined as:

$$
\Phi _ { k , \varepsilon } ^ { t } ( g ) = \sum _ { \ell = 1 } ^ { k } \sqrt { \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 } } , \quad g \in C .\tag{4}
$$

<sub>w</sub>h<sub>ere</sub> $\ell \in \{ 1 , 2 , \ldots , k \}$ i<sub>s</sub> th<sub>e</sub> d<sub>escen</sub>t i<sub>n</sub>d<sub>ex o</sub>f <sub>c</sub>li<sub>en</sub>t<sub>s.</sub>

![](images/695ce8705c9aa16de21d11e61d7eab2f8a1f12806697b4ab74ea18caf038219a.jpg)  
Figure 7: The gradient alignment result of FedAvg and GHA.

W<sub>e</sub> initi<sub>a</sub>liz<sub>e</sub> th<sub>e es</sub>tim<sub>a</sub>t<sub>e as</sub> $g \gets u _ { r _ { 1 } } ^ { t }$ and se<sub>q</sub>uentiall<sub>y</sub> incor-<sub>pora</sub>t<sub>e</sub> th<sub>e</sub> <sub>ran</sub>k<sub>e</sub>d di<sub>rec</sub>ti<sub>ons.</sub> U<sub>pon</sub> <sub>a</sub>ddi<sub>ng</sub> th<sub>e</sub> �<sub>-</sub>th di<sub>rec</sub>ti<sub>on,</sub> <sub>we</sub> <sub>p</sub>erform one warm-started Weiszfeld ste<sub>p</sub> usin<sub>g</sub> the current estimate $g$ (obtained from the <sub>p</sub>revious � − 1 ste<sub>p</sub>s) to u<sub>p</sub>date the centroid:

$$
w _ { \ell } ^ { ( k ) } = \frac { 1 } { \sqrt { \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 } } } ,\tag{5a}
$$

$$
g \gets \frac { \sum _ { \ell = 1 } ^ { k } w _ { \ell } ^ { ( k ) } u _ { r _ { \ell } } ^ { t } } { \sum _ { \ell = 1 } ^ { k } w _ { \ell } ^ { ( k ) } } ,\tag{5b}
$$

<sub>w</sub>h<sub>ere</sub> $\varepsilon > 0$ <sub>ensures</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>s</sub>t<sub>a</sub>bilit<sub>y.</sub> Aft<sub>er</sub> <sub>process</sub>i<sub>ng</sub> <sub>a</sub>ll <sub>ran</sub>k<sub>e</sub>d <sub>g</sub>radients<sub>,</sub> we obtain the reference direction b<sub>y</sub> normalizin<sub>g</sub> the fi<sub>na</sub>l <sub>es</sub>ti<sub>ma</sub>t<sub>e:</sub>

$$
u _ { \mathrm { r e f } } ^ { t } = \frac { g } { \left\| g \right\| } \quad ( \left\| g \right\| > 0 ) .\tag{6}
$$

This <sub>p</sub>ro<sub>g</sub>ressive construction strate<sub>gy</sub> confers robustness throu<sub>g</sub>h a dual mechanism: (i) the consistenc<sub>y</sub>-based rankin<sub>g</sub> ensures that the <sub>re</sub>f<sub>erence vec</sub>t<sub>or</sub> i<sub>s anc</sub>h<sub>ore</sub>d b<sub>y</sub> th<sub>e mos</sub>t <sub>re</sub>li<sub>a</sub>bl<sub>e gra</sub>di<sub>en</sub>t<sub>s</sub> d<sub>ur</sub>i<sub>ng</sub> the critical earl<sub>y p</sub>hases<sub>,</sub> thereb<sub>y</sub> miti<sub>g</sub>atin<sub>g</sub> initialization bias<sub>;</sub> and (ii) the Weiszfeld rewei<sub>g</sub>htin<sub>g</sub> naturall<sub>y</sub> attenuates the influence of u<sub>p</sub>dates with lar<sub>g</sub>e an<sub>g</sub>ular deviations (outliers), efectivel<sub>y</sub> s<sub>y</sub>nthesizin<sub>g</sub> a noise-resilient <sub>g</sub>lobal descent direction. Subse<sub>q</sub>uentl<sub>y,</sub> the <sub>server</sub> b<sub>roa</sub>d<sub>cas</sub>t<sub>s</sub> ${ u _ { \mathrm { r e f } } ^ { t } }$ to all <sub>p</sub>artici<sub>p</sub>atin<sub>g</sub> clients to orchestrate the <sub>g</sub>radient ali<sub>g</sub>nment in the forthcomin<sub>g</sub> GHA <sub>p</sub>hase.

## 4.3 Gradient Half-space Alignment

Th<sub>e re</sub>f<sub>erence gra</sub>di<sub>en</sub>t ${ \pmb u } _ { \mathrm { r e f } } ^ { t }$ <sub>co</sub>n<sub>s</sub>tr<sub>uc</sub>t<sub>e</sub>d b<sub>y</sub> RGPC <sub>se</sub>r<sub>ves as a</sub> r<sub>o</sub>- <sup>b</sup>ust proxy <sup>f</sup>or t<sup>h</sup>e g<sup>l</sup>o<sup>b</sup>a<sup>l</sup> optimization trajectory. In conventiona<sup>l</sup> a<sub>gg</sub>re<sub>g</sub>ation schemes such as FedAv<sub>g,</sub> the server indiscriminatel<sub>y</sub> avera<sub>g</sub>es client u<sub>p</sub>dates<sub>,</sub> ne<sub>g</sub>lectin<sub>g</sub> their directional dis<sub>p</sub>arities. As shown in Fi<sub>g</sub>. 7<sub>,</sub> this sim<sub>p</sub>le a<sub>gg</sub>re<sub>g</sub>ation allows conflictin<sub>g</sub> <sub>g</sub>radients to destructivel<sub>y</sub> interfere<sub>,</sub> leadin<sub>g</sub> to o<sub>p</sub>timization drift or <sub>cance</sub>ll<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>.</sub>

To miti<sub>g</sub>ate this<sub>,</sub> <sub>w</sub>e <sub>p</sub>ro<sub>p</sub>ose the GHA mechanism<sub>,</sub> <sub>w</sub>hich en-<sup>f</sup>orces <sup>d</sup>irectiona<sup>l</sup> agreement <sup>b</sup>y geometrica<sup>ll</sup>y projecting con<sup>fl</sup>icting <sub>gra</sub>di<sub>en</sub>t<sub>s on</sub>t<sub>o</sub> th<sub>e re</sub>f<sub>erence</sub> h<sub>a</sub>lf<sub>-space an</sub>d <sub>so</sub>ftl<sub>y a</sub>li<sub>gn</sub>i<sub>ng non</sub> o<sub>pp</sub>osin<sub>g</sub> <sub>up</sub>dates to<sub>w</sub>ard the reference direction. Therefore<sub>,</sub> GHA efectivel<sub>y</sub> neutralizes diver<sub>g</sub>ent com<sub>p</sub>onents while <sub>p</sub>reservin<sub>g</sub> the informative ortho<sub>g</sub>onal innovations. Formall<sub>y, g</sub>iven the server-<sub>s</sub>id<sub>e re</sub>f<sub>erence</sub> di<sub>rec</sub>ti<sub>on</sub> $\pmb { u } _ { \mathrm { r e f } } ^ { t }$ <sub>a</sub>t <sub>roun</sub>d �<sub>, we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e perm</sub>i<sub>ss</sub>ibl<sub>e</sub> <sub>re</sub>f<sub>erence</sub> h<sub>a</sub>lf<sub>-space:</sub>

$$
\mathcal { H } ^ { t } \triangleq \{ d \in \mathbb { R } ^ { d } : \langle d , u _ { \mathrm { r e f } } ^ { t } \rangle \geq 0 \} ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> <sub>a</sub>ll di<sub>rec</sub>ti<sub>ons</sub> th<sub>a</sub>t f<sub>orm</sub> <sub>a</sub> <sub>non-o</sub>bt<sub>use</sub> <sub>ang</sub>l<sub>e</sub> <sub>w</sub>ith $u _ { \mathrm { r e f } } ^ { t }$ <sub>.</sub> F<sub>or eac</sub>h <sub>c</sub>li<sub>en</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e</sub> $g _ { i } ^ { t } { \mathrm { : } }$ <sub>,</sub> GHA a<sub>pp</sub>lies a <sub>p</sub>iece<sub>w</sub>ise ali<sub>g</sub>nment <sub>ru</sub>l<sub>e:</sub>

$$
\hat { g } _ { i } ^ { t } = \{ \begin{array} { l l } { ( 1 - \lambda _ { i } ^ { t } ) g _ { i } ^ { t } + \lambda _ { i } ^ { t }  g _ { i } ^ { t }  u _ { \mathrm { r e f } } ^ { t } , } & { \langle g _ { i } ^ { t } , u _ { \mathrm { r e f } } ^ { t } \rangle \geq 0 , } \\ {  g _ { i } ^ { t } - \langle g _ { i } ^ { t } , u _ { \mathrm { r e f } } ^ { t } \rangle u _ { \mathrm { r e f } } ^ { t } , } & { \langle g _ { i } ^ { t } , u _ { \mathrm { r e f } } ^ { t } \rangle < 0 , } \end{array}\tag{7}
$$

Algorithm 1 PSL-RGHA   
1: Input: Client gradients $\{ g _ { i } ^ { t } \} _ { i \in { \mathcal { M } } } ,$ <sub>,</sub> smoothin<sub>g</sub> $\varepsilon > 0$   
2: Output: Reference gradient $u _ { \mathrm { r e f } } ^ { t } ,$ <sub>correc</sub>t<sub>e</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es</sub> $\{ \hat { g } _ { i } ^ { t }$ }<sub>�∈M</sub>   
3: Phase 1: Consistency scoring and ranking.   
4: for each $i \in \mathcal { M }$ do   
$\langle g _ { i } ^ { t } , g _ { j } ^ { t } \rangle$   
5: $\begin{array} { r } { S _ { i } ^ { t } \gets \frac { 1 } { | { \cal M } | - 1 } \sum _ { j \in { \cal M } , j \ne i } } \end{array}$   
∥<sub>�</sub><sup>�</sup><sub>�</sub> ∥·∥<sub>�</sub><sup>�</sup><sub>�</sub> ∥   
6: end for   
7<sub>:</sub> S<sub>o</sub>rt t<sub>o o</sub>bt<sub>a</sub>in $( r _ { 1 } , \ldots , r _ { M } )$ <sub>suc</sub>h th<sub>a</sub>t $S _ { r _ { 1 } } ^ { t } \geq \cdots \geq S _ { r _ { M } } ^ { t }$   
8: Phase 2: Progressive construction via Weiszfeld updates.   
9: $g  u _ { r _ { 1 } } ^ { t } \{ u _ { i } ^ { t } \triangleq g _ { i } ^ { t } / \Vert g _ { i } ^ { t } \Vert \}$   
10: for $k = 2$ to � do   
11: $w _ { \ell } ^ { ( k ) } \gets \big ( \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 } \big ) ^ { - 1 / 2 }$ f<sub>or</sub> $\ell = 1 , \ldots , k$   
12: $g \gets \frac { \sum _ { \ell = 1 } ^ { k } w _ { \ell } ^ { ( k ) } u _ { r _ { \ell } } ^ { t } } { \kappa }$   
${ \boldsymbol { \Sigma } } _ { \ell = 1 } ^ { k } { \boldsymbol { w } } _ { \ell } ^ { ( k ) }$   
13: end for   
14: $u _ { \mathrm { r e f } } ^ { t }  g / \| g \|$   
15: Phase 3: Gradient half-space alignment.   
16: for each client $i \in M$ in parallel do   
17: $s \gets \langle g _ { i } ^ { t } , u _ { \mathrm { r e f } } ^ { t } \rangle$   
18: $\textbf { i f } s \ge 0$ then   
19: $\begin{array} { r } { \lambda  \frac { 1 } { 2 } ( 1 - \frac { s } { \| g _ { i } ^ { t } \| } ) } \end{array}$   
20: $\hat { g } _ { i } ^ { t } \gets ( \dot { 1 } - \lambda ) g _ { i } ^ { t } + \lambda$ ∥<sub>�</sub><sup>�</sup><sub>�</sub> ∥ $u _ { \mathrm { r e f } } ^ { t }$   
21: else   
22: $\hat { g } _ { i } ^ { t }  g _ { i } ^ { t } - s \cdot u _ { \mathrm { r e f } } ^ { t }$   
23: end if   
24: end for   
25: return $u _ { \mathrm { r e f } } ^ { t } , \{ \hat { g } _ { i } ^ { t } \} _ { i \in \mathcal { M } . }$

where softl<sub>y</sub> ali<sub>g</sub>ns non-o<sub>pp</sub>osin<sub>g</sub> u<sub>p</sub>dates toward the reference ra<sub>y</sub> (while <sub>p</sub>reservin<sub>g</sub> the u<sub>p</sub>date norm), and <sub>p</sub>erforms the Euclidean projection onto $\bar { \mathcal { H } } ^ { t }$ <sup>to</sup> <sup>remove</sup> <sup>onl</sup>y <sup>the</sup> <sup>o</sup>pp<sup>osin</sup>g <sup>com</sup>p<sup>onent</sup> <sup>alon</sup>g ${ u _ { \mathrm { r e f } } ^ { t } }$ while <sub>p</sub>reservin<sub>g</sub> ortho<sub>g</sub>onal information.

Geometricall<sub>y</sub>, E<sub>q</sub>. (7) ali<sub>g</sub>ns each client u<sub>p</sub>date with the reference half-s<sub>p</sub>ace. For non-conflictin<sub>g</sub> u<sub>p</sub>dates (Case 1), GHA softl<sub>y</sub> <sub>pu</sub>ll<sub>s</sub> $g _ { i } ^ { t }$ t<sub>owar</sub>d th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> <sub>ray</sub> $| | g _ { i } ^ { t } | | u _ { \mathrm { r e f } } ^ { t } ,$ <sub>w</sub>ith <sub>sma</sub>ll<sub>er per</sub>t<sub>ur</sub>b<sub>a-</sub> tions for more consistent u<sub>p</sub>dates. For conflictin<sub>g</sub> u<sub>p</sub>dates (Case 2), it removes onl<sub>y</sub> the ne<sub>g</sub>ative com<sub>p</sub>onent alon<sub>g</sub> ${ \boldsymbol u _ { \mathrm { r e f } } ^ { t } }$ while <sub>p</sub>reservin<sub>g</sub> th<sub>e or</sub>th<sub>ogona</sub>l <sub>componen</sub>t<sub>.</sub> Th<sub>us, every correc</sub>t<sub>e</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\langle \hat { g } _ { i } ^ { t } , u _ { \mathrm { r e f } } ^ { t } \rangle \geq 0 .$ <sub>,</sub> su<sub>pp</sub>ortin<sub>g</sub> the <sub>g</sub>lobal descent direction without discardin<sub>g</sub> client-s<sub>p</sub>ecific information. GHA re<sub>q</sub>uires onl<sub>y</sub> �(�) vector <sub>opera</sub>ti<sub>ons</sub> <sub>an</sub>d i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> <sub>no</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>commun</sub>i<sub>ca</sub>ti<sub>on.</sub> Th<sub>e</sub> f<sub>u</sub>ll <sub>p</sub>rocedure is <sub>g</sub>iven in Al<sub>g</sub>orithm 1.

## 4.4 Theoretical Analysis

We <sub>p</sub>rovide theoretical insi<sub>g</sub>hts for the two ke<sub>y</sub> modules of PSL-RGHA. For RGPC<sub>,</sub> <sub>w</sub>e demonstrate that <sub>p</sub>ro<sub>g</sub>ressi<sub>v</sub>el<sub>y</sub> incor<sub>p</sub>oratin<sub>g</sub> <sub>ran</sub>k<sub>e</sub>d <sub>c</sub>li<sub>en</sub>t di<sub>rec</sub>ti<sub>ons—an</sub>d <sub>per</sub>f<sub>orm</sub>i<sub>ng a warm-s</sub>t<sub>ar</sub>t<sub>e</sub>d W<sub>e</sub>i<sub>sz</sub>f<sub>e</sub>ld re<sup>fi</sup>nement a<sup>f</sup>ter eac<sup>h</sup> step—<sup>f</sup>o<sup>ll</sup>ows a Majorization-Minimization (MM) framework over the convex set C. Conse<sub>q</sub>uentl<sub>y</sub>, the smoothed geometric-me<sup>d</sup>ian o<sup>b</sup>jective $\Phi _ { k , \varepsilon } ^ { t } ( \cdot )$ <sub>mono</sub>t<sub>on</sub>i<sub>ca</sub>ll<sub>y</sub> d<sub>ecreases a</sub>ft<sub>er</sub> each refinement. This <sub>g</sub>uarantees that RGPC reliabl<sub>y</sub> constructs <sub>a</sub> <sub>ro</sub>b<sub>us</sub>t <sub>re</sub>f<sub>erence</sub> <sub>gra</sub>di<sub>en</sub>t di<sub>rec</sub>ti<sub>on</sub> $u _ { \mathrm { r e f } } ^ { t }$ t<sub>o</sub> <sub>proxy</sub> th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l d<sub>e-</sub> <sub>scen</sub>t t<sub>ren</sub>d<sub>.</sub> B<sub>u</sub>ildi<sub>ng</sub> <sub>on</sub> thi<sub>s</sub> <sub>re</sub>f<sub>erence</sub> di<sub>rec</sub>ti<sub>on,</sub> <sub>we</sub> f<sub>ur</sub>th<sub>er</sub> <sub>ana</sub>l<sub>yze</sub> t<sup>h</sup>e geometric properties an<sup>d</sup> sta<sup>b</sup>i<sup>l</sup>ity o<sup>f</sup> GHA. GHA projects confli<sub>c</sub>ti<sub>ng</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>on</sub>t<sub>o</sub> th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> h<sub>a</sub>lf<sub>-space</sub> <sub>an</sub>d <sub>so</sub>ftl<sub>y</sub> <sub>a</sub>li<sub>gns</sub> <sub>non-con</sub>fli<sub>c</sub>ti<sub>ng</sub> <sub>ones</sub> t<sub>owar</sub>d $u _ { \mathrm { r e f } } ^ { t } .$ Th<sub>eorem</sub> 4<sub>.</sub>2 <sub>proves</sub> th<sub>a</sub>t th<sub>ese</sub> <sub>correc</sub>t<sub>e</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> <sub>res</sub>id<sub>e</sub> i<sub>n</sub> th<sub>e</sub> <sub>va</sub>lid h<sub>a</sub>lf<sub>-space,</sub> i<sub>mprove</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>a</sub>li<sub>gnmen</sub>t<sub>, an</sub>d <sub>rema</sub>i<sub>n non-expans</sub>i<sub>ve</sub> i<sub>n norm, w</sub>hi<sub>c</sub>h f<sub>un</sub>d<sub>a-</sub> mentall<sub>y</sub> ensures trainin<sub>g</sub> stabilit<sub>y</sub> under hetero<sub>g</sub>eneous <sub>g</sub>radient di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub>

Theorem 4.1 (MM descent of a progressive Weiszfeld step). Fix a training round � and a prefix size �. Let the feasible set be $C \triangleq \{ g \in \mathbb { R } ^ { d } : \| g \| \leq 1 \}$ and the smoothing parameter be $\varepsilon > 0$ . We define the smoothed geometric-median objective as:

$$
\Phi _ { k , \varepsilon } ^ { t } ( g ) \triangleq \sum _ { \ell = 1 } ^ { k } { \sqrt { \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 } } } , \qquad g \in C .\tag{8}
$$

Given a current estimate ${ \boldsymbol g } ^ { ( 0 ) } \in C ,$ , we define the distance and corresponding Weiszfeld weights $f o r \ell = 1 , . . . , k$ as:

$$
d _ { \ell } \triangleq \sqrt { \| g ^ { ( 0 ) } - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 } } , \quad w _ { \ell } \triangleq \frac { 1 } { d _ { \ell } } .\tag{9}
$$

Applying a single Weiszfeld update step yields:

$$
g ^ { ( 1 ) } \triangleq \frac { \sum _ { \ell = 1 } ^ { k } w _ { \ell } u _ { r _ { \ell } } ^ { t } } { \sum _ { \ell = 1 } ^ { k } w _ { \ell } } .\tag{10}
$$

Then, the updated estimate remains strictly within the convex set $( g ^ { ( 1 ) } \in C )$ , and the objective function monotonically decreases:

$$
\Phi _ { k , \varepsilon } ^ { t } ( g ^ { ( 1 ) } ) \leq \Phi _ { k , \varepsilon } ^ { t } ( g ^ { ( 0 ) } ) .\tag{11}
$$

Proof. The core of this <sub>p</sub>roof relies on the MM <sub>p</sub>rinci<sub>p</sub>le. We procee<sup>d</sup> in t<sup>h</sup>ree <sup>l</sup>ogica<sup>l</sup> steps: constructing a surrogate majorizer f<sub>unc</sub>ti<sub>on,</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> thi<sub>s</sub> <sub>surroga</sub>t<sub>e,</sub> <sub>an</sub>d <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> th<sub>e</sub> fi<sub>na</sub>l d<sub>e-</sub> <sub>scen</sub>t <sub>c</sub>h<sub>a</sub>i<sub>n.</sub>

We utilize the al<sub>g</sub>ebraic ine<sub>q</sub>ualit<sub>y</sub> $\begin{array} { r } { \sqrt { z } \le \frac { z } { 2 a } + \frac { a } { 2 } } \end{array}$ (valid for an<sub>y</sub> $z \geq 0$ <sub>an</sub>d <sub>�</sub> $~ > 0 ,$ with e<sub>q</sub>ualit<sub>y</sub> if and onl<sub>y</sub> if $a = { \sqrt { z } } )$ . B<sub>y</sub> s<sub>u</sub>bstit<sub>u</sub>tin<sub>g</sub> $z = \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 }$ <sub>an</sub>d $a = d _ { \ell } \geq \varepsilon > 0$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> <sub>a</sub> l<sub>oca</sub>li<sub>ze</sub>d <sub>upper</sub> <sup>b</sup>oun<sup>d f</sup>or eac<sup>h</sup> term in t<sup>h</sup>e o<sup>b</sup>jective:

$$
\sqrt { \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 } } \leq \frac { \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 } } { 2 d _ { \ell } } + \frac { d _ { \ell } } { 2 } .
$$

S<sub>umm</sub>i<sub>ng</sub> th<sub>ese</sub> b<sub>oun</sub>d<sub>s over a</sub>ll $\ell = 1 , \ldots , k$ <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> th<sub>e surroga</sub>t<sub>e</sub> majorizing <sup>f</sup>unction $Q ( g \mid g ^ { ( 0 ) } )$ :

$$
Q ( g \mid g ^ { ( 0 ) } ) \triangleq \sum _ { \ell = 1 } ^ { k } \left( { \frac { \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } + \varepsilon ^ { 2 } } { 2 d _ { \ell } } } + { \frac { d _ { \ell } } { 2 } } \right) .
$$

By construction, t<sup>h</sup>is surrogate strict<sup>l</sup>y upper-<sup>b</sup>oun<sup>d</sup>s our o<sup>b</sup>jective f<sub>unc</sub>ti<sub>on</sub> $\Phi _ { k . \varepsilon } ^ { t } ( g ) \leq Q ( g \mid g ^ { ( 0 ) } )$ f<sub>or a</sub>ll $g \in C ,$ <sub>,</sub> <sub>an</sub>d ti<sub>g</sub>ht<sub>ens</sub> <sub>exac</sub>tl<sub>y</sub> <sub>a</sub>t th<sub>e curren</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>e:</sub> $Q ( g ^ { ( 0 ) } \mid g ^ { ( 0 ) } ) = \Phi _ { k , \varepsilon } ^ { t } ( g ^ { ( 0 ) } )$ .

Next<sub>,</sub> we find the o<sub>p</sub>timal u<sub>p</sub>date b<sub>y</sub> minimizin<sub>g</sub> $Q ( { \bf \cdot } \mid g ^ { ( 0 ) } )$ over the feasible set C. Discardin<sub>g</sub> constants inde<sub>p</sub>endent of <sub>�</sub>, this is mathematicall<sub>y</sub> e<sub>q</sub>uivalent to minimizin<sub>g</sub> the wei<sub>g</sub>hted leasts<sub>q</sub>uares function:

$$
J ( g ) \triangleq \frac 1 2 \sum _ { \ell = 1 } ^ { k } w _ { \ell } \| g - u _ { r _ { \ell } } ^ { t } \| ^ { 2 } , \qquad g \in C .
$$

Let $\begin{array} { r } { \alpha \triangleq \sum _ { \ell = 1 } ^ { k } w _ { \ell } > 0 } \end{array}$ . B<sub>y</sub> com<sub>p</sub>letin<sub>g</sub> the s<sub>q</sub>uare<sub>,</sub> we can rewrite $J ( g )$ as:

$$
J ( \boldsymbol { g } ) = \frac { \alpha } { 2 } \lVert \boldsymbol { g } - \tilde { \boldsymbol { g } } ^ { ( 1 ) } \rVert ^ { 2 } + \mathrm { c o n s t . } \quad \mathrm { w h e r e } \quad \tilde { \boldsymbol { g } } ^ { ( 1 ) } = \frac { \sum _ { \ell = 1 } ^ { k } w _ { \ell } u _ { r _ { \ell } } ^ { t } } { \sum _ { \ell = 1 } ^ { k } w _ { \ell } } .
$$

Sin<sub>ce eac</sub>h $w _ { \ell } ~ > ~ 0$ <sub>an</sub>d $\| u _ { r _ { \ell } } ^ { t } \| = 1$ <sub>,</sub> th<sub>e vec</sub>t<sub>or</sub> $\tilde { g } ^ { ( 1 ) }$ i<sub>s a co</sub>n<sub>ve</sub>x combination of <sub>p</sub>oints on the unit s<sub>p</sub>here. Conse<sub>q</sub>uentl<sub>y,</sub> $\| \tilde { g } ^ { ( 1 ) } \| \leq 1 .$ mean<sup>i</sup>n<sub>g</sub> $\tilde { g } ^ { ( 1 ) }$ alread<sub>y</sub> lies <sub>p</sub>erfectl<sub>y</sub> within our constraint set C (the constraint is inactive). Therefore, the minimizer over $c$ i<sub>s</sub> exact<sup>l</sup><sub>y</sub> $g ^ { ( 1 ) } = \tilde { g } ^ { ( 1 ) }$ <sub>.</sub> B<sub>ecause</sub> $g ^ { ( 1 ) }$ minimizes the majorizer �, we can <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>e</sub> fi<sub>na</sub>l d<sub>escen</sub>t <sub>c</sub>h<sub>a</sub>i<sub>n:</sub>

$$
\Phi _ { k , \varepsilon } ^ { t } ( g ^ { ( 1 ) } ) \leq Q ( g ^ { ( 1 ) } \mid g ^ { ( 0 ) } ) \leq Q ( g ^ { ( 0 ) } \mid g ^ { ( 0 ) } ) = \Phi _ { k , \varepsilon } ^ { t } ( g ^ { ( 0 ) } ) .
$$

Thi<sub>s con</sub>fi<sub>rms</sub> th<sub>e mono</sub>t<sub>on</sub>i<sub>c</sub> d<sub>escen</sub>t <sub>proper</sub>t<sub>y an</sub>d <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> <sub>p</sub>roo<sup>f</sup>. □

Establishin<sub>g</sub> Theorem 4.1 that RGPC reliabl<sub>y</sub> constr<sub>u</sub>cts a rob<sub>u</sub>st <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l d<sub>escen</sub>t di<sub>rec</sub>ti<sub>on,</sub> <sub>we</sub> <sub>now</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> h<sub>ow</sub> GHA <sub>u</sub>tili<sub>zes</sub> thi<sub>s</sub> <sub>re</sub>f<sub>erence</sub> t<sub>o</sub> <sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub> h<sub>e</sub>t<sub>erogeneous</sub> <sub>c</sub>li<sub>en</sub>t <sub>up</sub>d<sub>a</sub>t<sub>es.</sub> Th<sub>eo-</sub> rem 4.2 proves t<sup>h</sup>at t<sup>h</sup>e GHA projection not on<sup>l</sup>y guarantees strict di<sub>rec</sub>ti<sub>ona</sub>l <sub>a</sub>li<sub>gnmen</sub>t <sub>w</sub>ith th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l t<sub>ren</sub>d<sub>,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> i<sub>n</sub>h<sub>eren</sub>tl<sub>y</sub> <sub>en-</sub> sures numerical stabilit<sub>y</sub> b<sub>y</sub> <sub>p</sub>reventin<sub>g</sub> <sub>g</sub>radient norm ex<sub>p</sub>ansion:

Theorem 4.2 (Geometric meaning and stability of GHA). Let � $\epsilon \mathbb { R } ^ { d }$ be a unit vector and define the reference half-space $\mathcal { H } \triangleq$ $\{ d \in \mathbb { R } ^ { d } : \langle d , u \rangle \geq 0 \}$ . For any $g \in \mathbb { R } ^ { d }$ , let $\hat { \boldsymbol g }$ be given by (7). Then:

(1) (Half-space feasibility and monotone reference projection) $\langle \hat { g } , u \rangle \geq 0 \ a n d \langle \hat { g } , u \rangle \geq \langle g , u \rangle .$

(2) (Minimum change under conflict) $I f \left. g , u \right. \ < \ 0 _ { ; }$ , then ${ \hat { g } } =$ $\Pi _ { \mathcal { H } } ( g )$ and thus $\begin{array} { r } { \| \hat { g } - g \| = \operatorname* { m i n } _ { d \in \mathcal { H } } \| d - g \| } \end{array}$

(3) (Non-expansiveness in norm) $\| \hat { g } \| \leq \| g \|$

Proof<sub>.</sub> L<sub>e</sub>t $s \triangleq \langle g , u \rangle$ <sup>b</sup>e t<sup>h</sup>e projection sca<sup>l</sup>ar. We ana<sup>l</sup>yze t<sup>h</sup>e <sub>a</sub>li<sub>gnmen</sub>t b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> <sub>un</sub>d<sub>er</sub> t<sub>wo</sub> <sub>mu</sub>t<sub>ua</sub>ll<sub>y</sub> <sub>exc</sub>l<sub>us</sub>i<sub>ve</sub> <sub>scenar</sub>i<sub>os</sub> b<sub>ase</sub>d on the <sub>g</sub>radient conflict. Accordin<sub>g</sub> to the <sub>p</sub>iecewise ali<sub>g</sub>nment rule in (7), we have $\hat { g } = g - s u$ . Conse<sub>q</sub>uentl<sub>y,</sub> the inner <sub>p</sub>roduct with th<sub>e re</sub>f<sub>erence</sub> di<sub>rec</sub>ti<sub>on</sub> b<sub>ecomes:</sub>

$$
\left. \hat { g } , u \right. = s - s = 0 .
$$

This im<sub>p</sub>lies both $\langle \hat { g } , u \rangle \geq 0$ <sub>an</sub>d $\langle \hat { g } , u \rangle \geq s ,$ , satisf<sub>y</sub>in<sub>g</sub> condition (1). F<sub>u</sub>rth<sub>e</sub>rm<sub>o</sub>r<sub>e, s</sub>in<sub>ce</sub> $\hat { g } \perp u$ <sub>an</sub>d th<sub>e or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>up</sub>d<sub>a</sub>t<sub>e can</sub> b<sub>e</sub> d<sub>ecom-</sub> <sub>p</sub>ose<sup>d</sup> as $\boldsymbol { g } = \boldsymbol { \hat { g } } + s \boldsymbol { u } ,$ <sub>,</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the P<sub>y</sub>tha<sub>g</sub>orean theorem <sub>y</sub>ields:

$$
\| g \| ^ { 2 } = \| \hat { g } \| ^ { 2 } + s ^ { 2 } \quad \Rightarrow \quad \| \hat { g } \| \leq \| g \| .
$$

Regarding the projection claim (2), the Euclidean projection onto the half-s<sub>p</sub>ace H satisfies $\Pi _ { \mathcal { H } } ( g ) = g \operatorname { i f } s \geq 0 ,$ <sub>,</sub> <sub>an</sub>d $\Pi _ { \mathcal { H } } ( g ) = g - s u \operatorname { i f } s <$ 0<sub>.</sub> Th<sub>us,</sub> $\hat { g } = \Pi _ { \mathcal { H } } ( g )$ <sub>,</sub> and the minimum-distance <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> naturall<sub>y</sub> <sup>f</sup>o<sup>ll</sup>ows <sup>f</sup>rom t<sup>h</sup>e stan<sup>d</sup>ar<sup>d</sup> <sup>d</sup>e<sup>fi</sup>nition o<sup>f</sup> Euc<sup>l</sup>i<sup>d</sup>ean projections.

Let $\lambda \in [ 0 , 1 / 2 ]$ be the ali<sub>g</sub>nment coeficient defined in (7), such th<sub>a</sub>t th<sub>e so</sub>ftl<sub>y a</sub>li<sub>gne</sub>d <sub>gra</sub>di<sub>en</sub>t i<sub>s</sub> $\begin{array} { r } { \hat { g } = ( 1 - \lambda ) g + \lambda \| g \| u . } \end{array}$ <sub>.</sub> Th<sub>e</sub> i<sub>nner</sub> <sub>pro</sub>d<sub>uc</sub>t <sub>w</sub>ith th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> di<sub>rec</sub>ti<sub>on</sub> <sub>expan</sub>d<sub>s</sub> <sub>as:</sub>

$$
\begin{array} { l } { \langle \hat { g } , u \rangle = ( 1 - \lambda ) s + \lambda \| g \| } \\ { \qquad \geq ( 1 - \lambda ) s + \lambda s = s \geq 0 , } \end{array}
$$

which <sub>p</sub>roves condition (1) for this case.

To establish the non-ex<sub>p</sub>ansiveness <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> (3), we introduce auxiliar<sub>y</sub> variables $a \triangleq g$ <sub>an</sub>d $b \triangleq \Vert g \Vert u$ <sub>.</sub> N<sub>o</sub>ti<sub>ce</sub> th<sub>a</sub>t $\left\| a \right\| = \left\| b \right\| =$ ∥<sub>�</sub>∥, and our ali<sub>g</sub>ned <sub>g</sub>radient can be written as the convex combin<sub>a</sub>ti<sub>o</sub>n $\hat { g } = ( 1 - \lambda ) a + \lambda b$ . Usin<sub>g</sub> the standard norm identit<sub>y</sub> for <sub>a</sub>fi<sub>ne com</sub>bi<sub>na</sub>ti<sub>ons:</sub>

$$
\| ( 1 - \lambda ) a + \lambda b \| ^ { 2 } = ( 1 - \lambda ) \| a \| ^ { 2 } + \lambda \| b \| ^ { 2 } - \lambda ( 1 - \lambda ) \| a - b \| ^ { 2 } ,
$$

<sub>we can su</sub>b<sub>s</sub>tit<sub>u</sub>t<sub>e � an</sub>d � b<sub>ac</sub>k t<sub>o o</sub>bt<sub>a</sub>i<sub>n:</sub>

$$
\| \hat { g } \| ^ { 2 } = \| g \| ^ { 2 } - \lambda ( 1 - \lambda ) \| g - \| g \| u \| ^ { 2 } \leq \| g \| ^ { 2 } .
$$

![](images/201d8a41a90f022a8a98a6b27d16a9721bc483858e5177f650e937b6a358e93c.jpg)  
(a) ResNet18 CIFAR10 (IID).

![](images/26a2c60d57d48d34ec7d028b9577fc491c13a5a13b813175fd9ec4774b75a861.jpg)  
(b) ResNet18 CIFAR100 (IID).

![](images/1fb9e6056dbb4af4d3f2ac9917ca375181ca9361e03e16760889357736f2e308.jpg)  
(c) ResNet18 FMNIST (IID).

![](images/9c6a2bf7f735e9850256683aaae62d1ade9e9daef68cc6f6d5762b520a30e8c5.jpg)  
<sup>Round</sup>(d) ResNet-32 CIFAR10 (IID).<sup>Round</sup>

![](images/ef37f7fa29ff6d8e96737a9af13e1daca65e23545b1dc31f8ac952d55d5bd055.jpg)

![](images/088afe9ffb5d6cb380507bc51c1e447aeb3bd7c092bc4ec059f3d1006e8464ff.jpg)

![](images/62af0027eed3ac48ce8cc0ebca6644933ef27e3d3911e8ff90fe5da5c668fcbe.jpg)

<sup>RoundRound</sup>(e) ResNet32 CIFAR100 (IID).  
![](images/231bce22d36876ea250caf86eb9acae877eea5664840b0781e0ca256aaa84cd3.jpg)

![](images/d4e6b6c35c47da218ae0664029ca086fd296881aa31c30e6faadfc9d0fa57447.jpg)

![](images/bdd0cf25a81c57d4d63768634f4d37bfd09f1090704dc148f609a5d36f4a57eb.jpg)

(f) ResNet32 FMNIST (IID).  
![](images/780091e4cc9cdc2c6d991fe6c17275f1fd6e0089d43a8308a9de564798ddfd30.jpg)

![](images/a1ca0c39d52bc999e81c175001ad729be7cecd41df5eff2573367e171d0caa84.jpg)  
Round   Round   Round   Round   Round   Round   <sup>0</sup> <sup>100</sup> <sup>200</sup> <sup>300 0</sup> <sup>100</sup> <sup>200</sup> <sup>300 0</sup> <sup>100</sup> <sup>2000</sup> <sup>100</sup> <sup>200</sup> <sup>300 0</sup> <sup>100</sup> <sup>200</sup> <sup>300 0</sup> <sup>100</sup> <sup>200</sup> <sup>30</sup>  (g) ResNet18 CIFAR10 (non-IID).(h) ResNet18CIFAR100(non-IID). (i) ResNet18 FMNIST (non-IID). (j) ResNet32 CIFAR10 (non-IID).(k) ResNet32CIFAR100(non-IID). (l) ResNet32 FMNIST (non-IID)

Figure 8: Performance of PSL-RGHA on FMNIST, CIFAR10, and CIFAR100 datasets under IID and non-IID settings.  
![](images/1242af6e43b96a00beeabf0d1a78b9b8a2449e303847837f4808d06f29c6e84f.jpg)  
(a) Speech commands (IID).

![](images/ac046cbd8421503a3c8135ad7e6b8288a034dea19f0a92df1e342cf815b10c7c.jpg)

![](images/6bf47ab1419828b3e9c61870a9c23ae453488f7a0174ee53925c63ceca993f9e.jpg)  
0 20 40 60 80 1000 20 40 60(c) Speech commands (non-IID).

(b) SNIPS (IID).  
![](images/d6c5df1043bfc8454892435283d84a5349ea2497a8a82a41492563ee1ab0480a.jpg)  
40 60 80 1000 20 40 60 8(d) SNIPS (non-IID).  
Figure 9: Performance of PSL-RGHA on Speech commands and SNIPS datasets under IID and non-IID settings.

Sin<sub>ce</sub> $\lambda ( 1 - \lambda ) \geq 0 \mathrm { f o r } \lambda \in [ 0 , 1 / 2 ] ,$ , this immediatel<sub>y</sub> <sub>y</sub>ields ∥<sub>�</sub>ˆ∥ ≤ ∥<sub>�</sub>∥. Combinin<sub>g</sub> both cases efectivel<sub>y</sub> covers the entire <sub>g</sub>radient s<sub>p</sub>ace, com<sub>p</sub><sup>l</sup>et<sup>i</sup>n<sub>g</sub> t<sup>h</sup>e <sub>p</sub>roo<sup>f</sup>. □

## 5 Experimental

## 5.1 Experimental Setup

T<sub>o</sub> <sub>co</sub>m<sub>p</sub>r<sub>e</sub>h<sub>e</sub>n<sub>s</sub>i<sub>ve</sub>l<sub>y</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e</sub> <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> <sub>o</sub>f PSL-RGHA<sub>,</sub> <sub>we</sub> utilize five standard benchmarks: FMNIST [50], CIFAR-10, CIFAR-100 [19], S<sub>p</sub>eech commands [44] and SNIPS [5] under both IID and non-IID settin<sub>g</sub>s. We em<sub>p</sub>lo<sub>y</sub> ResNet-18 and ResNet-32 [13] as the backbone architectures. To ali<sub>g</sub>n with the s<sub>p</sub>lit learnin<sub>g</sub> <sub>p</sub>aradi<sub>g</sub>m<sub>,</sub> each model is <sub>p</sub>artitioned at a s<sub>p</sub>ecific cut la<sub>y</sub>er<sub>,</sub> with the initial l<sub>ayers</sub> d<sub>ep</sub>l<sub>oye</sub>d <sub>on</sub> th<sub>e c</sub>li<sub>en</sub>t <sub>an</sub>d th<sub>e rema</sub>i<sub>n</sub>i<sub>ng</sub> l<sub>ayers on</sub> th<sub>e server.</sub> Th<sub>e</sub> tr<sub>a</sub>inin<sub>g e</sub>n<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt <sub>co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>s o</sub>f 10 <sub>c</sub>li<sub>e</sub>nt d<sub>ev</sub>i<sub>ces.</sub> T<sub>o s</sub>im<sub>u</sub>l<sub>a</sub>t<sub>e</sub> statistical hetero<sub>g</sub>eneit<sub>y</sub> (non-IID data), we <sub>p</sub>artition the trainin<sub>g</sub> data amon<sub>g</sub> clients usin<sub>g</sub> a Dirichlet distribution ���(�) [51], where the concentration <sub>p</sub>arameter � is varied across {0.1, 0.3, 0.5, 0.7} to <sub>represen</sub>t i<sub>ncreas</sub>i<sub>ng</sub> d<sub>egrees</sub> <sub>o</sub>f d<sub>a</sub>t<sub>a</sub> <sub>s</sub>k<sub>ew.</sub> All <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>are</sub> t<sub>ra</sub>i<sub>ne</sub>d b<sub>y</sub> the SGD [20] o<sub>p</sub>timizer.

## 5.2 Benchmarks

T<sub>o co</sub>m<sub>p</sub>r<sub>e</sub>h<sub>e</sub>n<sub>s</sub>i<sub>ve</sub>l<sub>y eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce o</sub>f PSL-RGHA<sub>, we</sub> com<sub>p</sub>are the <sub>p</sub>ro<sub>p</sub>osed PSL-RGHA a<sub>g</sub>ainst the followin<sub>g</sub> re<sub>p</sub>resentative baselines: i) SFL [39] is a h<sub>y</sub>brid architecture that inte<sub>g</sub>rates SL <sub>w</sub>ith FL<sub>.</sub> SFL <sub>e</sub>n<sub>a</sub>bl<sub>es</sub> <sub>pa</sub>r<sub>a</sub>ll<sub>e</sub>l <sub>c</sub>li<sub>e</sub>nt-<sub>s</sub>id<sub>e</sub> tr<sub>a</sub>inin<sub>g</sub> <sub>ac</sub>r<sub>oss</sub> <sub>c</sub>li<sub>e</sub>nt d<sub>ev</sub>i<sub>ces, w</sub>h<sub>ere eac</sub>h d<sub>ev</sub>i<sub>ce up</sub>d<sub>a</sub>t<sub>es</sub> it<sub>s c</sub>li<sub>en</sub>t<sub>-s</sub>id<sub>e mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>up-</sub> loads it to the <sub>p</sub>arameter server for model a<sub>gg</sub>re<sub>g</sub>ation. ii) PSL [15] i<sub>s</sub> <sub>a</sub> <sub>s</sub>tr<sub>ea</sub>mlin<sub>e</sub>d <sub>va</sub>ri<sub>a</sub>nt <sub>o</sub>f SFL th<sub>a</sub>t <sub>e</sub>limin<sub>a</sub>t<sub>es</sub> <sub>c</sub>li<sub>e</sub>nt-<sub>s</sub>id<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l a<sub>gg</sub>re<sub>g</sub>ation. PSL trains a shared server-side model usin<sub>g</sub> activati<sub>ons</sub> f<sub>rom</sub> <sub>a</sub>ll <sub>c</sub>li<sub>en</sub>t<sub>s,</sub> <sub>w</sub>hil<sub>e</sub> <sub>eac</sub>h <sub>c</sub>li<sub>en</sub>t i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> it<sub>s</sub> local sub-model without cross-device coordination. iii) EPSL [32] is o<sub>p</sub>timized PSL <sub>v</sub>ariant that red<sub>u</sub>ces com<sub>pu</sub>tational and comm<sub>u</sub>- nication burdens b<sub>y</sub> a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> onl<sub>y</sub> a fraction of the cut-la<sub>y</sub>er <sub>gra</sub>di<sub>en</sub>t<sub>s</sub> <sub>on</sub> th<sub>e</sub> <sub>server</sub> <sub>s</sub>id<sub>e,</sub> <sub>prov</sub>idi<sub>ng</sub> <sub>a</sub> <sub>m</sub>iddl<sub>e</sub> <sub>groun</sub>d b<sub>e</sub>t<sub>ween</sub> f<sub>u</sub>ll <sub>aggrega</sub>ti<sub>on</sub> <sub>an</sub>d t<sub>o</sub>t<sub>a</sub>l i<sub>n</sub>d<sub>epen</sub>d<sub>ence.</sub>

Table 1: Test accuracy (%) of PSL-RGHA and baselines under IID and non-IID settings.
<table><tr><td>Dataset</td><td>Setting</td><td>Backbone</td><td>PSL</td><td>SFL</td><td>EPSL</td><td>PSL-RGHA</td></tr><tr><td rowspan="4">CIFAR10</td><td rowspan="2">IID</td><td>ResNet-18</td><td>78.36</td><td>83.71</td><td>84.62</td><td>84.98</td></tr><tr><td>ResNet-32</td><td>83.71</td><td>86.11</td><td>86.41</td><td>87.66</td></tr><tr><td rowspan="2">non-IID</td><td>ResNet-18</td><td>52.32</td><td>79.74</td><td>67.01</td><td>82.68</td></tr><tr><td>ResNet-32</td><td>70.26</td><td>75.08</td><td>75.61</td><td>79.17</td></tr><tr><td rowspan="4">CIFAR100</td><td rowspan="2">IID</td><td>ResNet-18</td><td>56.42</td><td>46.52</td><td>54.08</td><td>59.28</td></tr><tr><td>ResNet-32</td><td>58.04</td><td>59.88</td><td>58.62</td><td>60.23</td></tr><tr><td rowspan="2">non-IID</td><td>ResNet-18</td><td>44.87</td><td>39.71</td><td>45.71</td><td>47.52</td></tr><tr><td>ResNet-32</td><td>46.14</td><td>47.61</td><td>46.84</td><td>49.58</td></tr><tr><td rowspan="4">FMNIST</td><td rowspan="2">IID</td><td>ResNet-18</td><td>89.17</td><td>90.95</td><td>89.21</td><td>93.49</td></tr><tr><td>ResNet-32</td><td>87.72</td><td>89.52</td><td>90.41</td><td>91.25</td></tr><tr><td rowspan="2">non-IID</td><td>ResNet-18</td><td>78.47</td><td>80.42</td><td>81.36</td><td>86.83</td></tr><tr><td>ResNet-32</td><td>72.89</td><td>82.37</td><td>81.55</td><td>83.05</td></tr></table>

## 5.3 Performance Evaluation

5.3.1 Superiority ofPSL-RGHA. We first evaluate the convergence <sub>a</sub>nd <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce o</sub>f PSL-RGHA<sub>.</sub> Fi<sub>g.</sub> 8 <sub>a</sub>nd Fi<sub>g.</sub> 9 <sub>s</sub>h<sub>ow</sub> th<sub>e</sub> t<sub>es</sub>t <sub>accuracy</sub> <sub>curves</sub> <sub>on</sub> <sub>v</sub>i<sub>s</sub>i<sub>on</sub> <sub>an</sub>d <sub>speec</sub>h t<sub>as</sub>k<sub>s,</sub> <sub>respec-</sub> ti<sub>ve</sub>l<sub>y, w</sub>hil<sub>e</sub> T<sub>a</sub>bl<sub>e</sub> 1 <sub>repor</sub>t<sub>s</sub> th<sub>e</sub> fi<sub>na</sub>l t<sub>op-</sub>1 <sub>accuracy un</sub>d<sub>er</sub> dif<sub>er-</sub> ent datasets and hetero<sub>g</sub>eneit<sub>y</sub> settin<sub>g</sub>s. PSL-RGHA consistentl<sub>y</sub> <sub>ac</sub>hi<sub>eves</sub> f<sub>as</sub>t<sub>er</sub> <sub>convergence</sub> <sub>an</sub>d hi<sub>g</sub>h<sub>er</sub> <sub>accuracy</sub> th<sub>an</sub> <sub>a</sub>ll b<sub>ase</sub>li<sub>nes</sub> across vision<sub>,</sub> ke<sub>y</sub>word s<sub>p</sub>ottin<sub>g,</sub> and s<sub>p</sub>eech classification tasks. This <sub>g</sub>ain mainl<sub>y</sub> comes from its <sub>g</sub>radient <sub>g</sub>uidance and correction <sub>mec</sub>h<sub>an</sub>i<sub>sm.</sub> U<sub>n</sub>lik<sub>e van</sub>ill<sub>a</sub> PSL<sub>, w</sub>hi<sub>c</sub>h <sub>su</sub>f<sub>ers</sub> f<sub>rom severe c</sub>li<sub>en</sub>t <sup>d</sup>ri<sup>f</sup>t, PSL-RGHA uses GHA to project con<sup>fl</sup>icting gra<sup>d</sup>ients onto t<sup>h</sup>e <sub>re</sub>f<sub>erence</sub> h<sub>a</sub>lf<sub>-space, ensur</sub>i<sub>ng cons</sub>t<sub>ruc</sub>ti<sub>ve up</sub>d<sub>a</sub>t<sub>es an</sub>d <sub>re</sub>d<sub>uc</sub>i<sub>ng</sub> oscillations caused b<sub>y</sub> data hetero<sub>g</sub>eneit<sub>y</sub>. EPSL onl<sub>y</sub> <sub>p</sub>erforms <sub>p</sub>arti<sub>a</sub>l <sub>aggrega</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> <sub>canno</sub>t f<sub>u</sub>ll<sub>y</sub> <sub>correc</sub>t <sub>gra</sub>di<sub>en</sub>t <sub>con</sub>fli<sub>c</sub>t<sub>s,</sub> resultin<sub>g</sub> in lower conver<sub>g</sub>ence <sub>p</sub>erformance.

5.3.2 Impact of Client Scalability. Fig. 10 illustrates the converged accurac<sub>y</sub> as the number of <sub>p</sub>artici<sub>p</sub>atin<sub>g</sub> clients varies from 10 to 40. PSL-RGHA sho<sub>w</sub>s s<sub>up</sub>erior scalabilit<sub>y,</sub> consistentl<sub>y</sub> o<sub>u</sub>t<sub>p</sub>erformin<sub>g</sub> b<sub>ase</sub>li<sub>nes</sub> <sub>across</sub> <sub>a</sub>ll d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> <sub>an</sub>d <sub>se</sub>tti<sub>ngs.</sub> I<sub>n</sub> b<sub>a</sub>l<sub>ance</sub>d IID <sub>env</sub>i<sub>ron-</sub> <sub>men</sub>t<sub>s, a</sub>ll <sub>me</sub>th<sub>o</sub>d<sub>s rema</sub>i<sub>n re</sub>l<sub>a</sub>ti<sub>ve</sub>l<sub>y s</sub>t<sub>a</sub>bl<sub>e as</sub> th<sub>e sys</sub>t<sub>em sca</sub>l<sub>es.</sub>

Roundound <sup>0</sup> <sup>100</sup> <sup>200</sup><sub>Round</sub> <sup>100</sup> <sup>200</sup><sub>Round</sub> (d) CIFAR-10(non-IID).

![](images/962fa4749b502a297c43b5ccec3041d69f7222eac1bfe3ce7fe972e3efe0c885.jpg)  
(a) CIFAR-10 (IID).

![](images/adbd15b27246050728780be6e0428c16b90f4c607a6f12793836a471f5cc0cb3.jpg)  
(b) CIFAR-100 (IID).

![](images/94772aa7769af6e50312c505e38d9c0bdd1f1891e61dc5d657bf83ea0afe4340.jpg)

![](images/da68777655850392defaf94224b6cd044639de19c3c4361cac4c4debf0786627.jpg)

![](images/ad918d8eea43d06987bd6276c487cff88093cb206746856e12ca7bed1bf8d90b.jpg)  
20 3030 4030 40Number of Edge Devicesf Edge Devicesge Devic s Number of Edge Devicesmber of Edge Devices of Edge Devic s(d) CIFAR-10 (non-IID).  
10 2020 3020 30Number of ENumber of Edge Devicesmber of Edge Devic s Number of EdgNumber of Edge DevicesNumber of Edge Devic s(e) CIFAR-100 (non-IID).

(c) FMNIST (IID).  
![](images/bf44b2e1d7e0f858f47bf9e6d2f25de54d8ec6f53b55a9df4770f4ad45b872e2.jpg)  
10 200 20 3020 30Number Number of Edge DeNumber of Edge Devic s Number Number of Edge DeNumber of Edge Devic s(f) FMNIST (non-IID).

Figure 10: The impact of the number of edge devices on ResNet-18 test accuracy under IID and non-IID settings.90 50 9050 9050 90  
![](images/6a0241918589e761aa5a6a80e79efaa4e4c3f342243f659439ecea8ac6246aa8.jpg)  
(a) ResNet-18 CIFAR-10.

![](images/17bdbb3e18e381d364d99e53dc4db60c8e3b66022235e7d7e8b0648dfb494a3f.jpg)  
Data Heterog Heterogenei y ( )terogeneity ( ) (b) ResNet-18 CIFAR-100.

![](images/090097c9e076ee4a81b36b8317e78b33da2b1009a15fa5fd26a79769645a0ac0.jpg)  
Data HeteData Heterogenei yData Heterogeneity ( )(c) ResNet-18 FMNIST.

![](images/08b262aefd2034041600c223830cf417ee49942c0cc9ef6d20c55186a078e786.jpg)  
<sup>0.1</sup> <sup>0.3</sup> <sup>0.540</sup> <sup>0.3</sup> <sup>0.5</sup> <sup>0.7</sup> <sup>0.3</sup> <sup>0.5</sup> <sup>0.7Data</sup> <sup>Heterogeneity</sup> <sup>( )</sup>(d) ResNet-32 CIFAR-10.<sup>Data</sup> <sup>Heterogen i y</sup> <sup>(</sup> <sup>)</sup> <sup>Heterogeneity</sup> <sup>(</sup> <sup>)</sup>

![](images/ff4a2d664133471322874644445141f675a5806037c11ef3416661928e87dd5c.jpg)

![](images/eee78cbabb2fa72e2091c99928ffa185c6a0adf85d763749fe38ca1dda67e656.jpg)  
15 Data HeterogData Heterogen i y ( )ta Heterogeneity ( ) <sup>0.1</sup> <sup>0.30.1</sup> <sup>0.3</sup> <sup>0.51</sup> <sup>0.3</sup> <sup>0.5</sup> <sup>0.</sup>(e) ResNet-32 CIFAR-100.  
40T<sup>e</sup> Data HeteData Heterogen i yData Heterogeneity ( ) <sup>0.7 0.1</sup> <sup>0.30.1</sup> <sup>0.3</sup> <sup>0.51</sup> <sup>0.3</sup> <sup>0.5</sup>(f) ResNet-32 FMNIST.  
Figure 11: The impact of data heterogeneity on test accuracy under non-IID settings.

H<sub>oweve</sub>r<sub>,</sub> PSL-RGHA m<sub>a</sub>int<sub>a</sub>in<sub>s a</sub> di<sub>s</sub>tin<sub>c</sub>t <sub>pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce e</sub>d<sub>ge.</sub> Thi<sub>s</sub> stems <sup>f</sup>rom t<sup>h</sup>e GHA mo<sup>d</sup>u<sup>l</sup>e, w<sup>h</sup>ic<sup>h</sup> re<sup>fi</sup>nes t<sup>h</sup>e <sup>d</sup>escent trajectory <sub>even w</sub>h<sub>en</sub> d<sub>a</sub>t<sub>a</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons are un</sub>if<sub>orm, w</sub>hil<sub>e</sub> th<sub>e o</sub>th<sub>er</sub> b<sub>ase</sub>li<sub>nes</sub> are limited b<sub>y</sub> the inherent variance of avera<sub>g</sub>in<sub>g</sub> stochastic <sub>g</sub>radi ents. Under hetero<sub>g</sub>eneous settin<sub>g</sub>s<sub>,</sub> as the client scale increases<sub>,</sub> baselines sufer from si<sub>g</sub>nificant <sub>p</sub>erformance deterioration. This is attributed to the accumulation of <sub>g</sub>radient noise. With naive a<sub>gg</sub>re-<sub>g</sub>ation<sub>,</sub> addin<sub>g</sub> more hetero<sub>g</sub>eneous clients introduces more conflicti<sub>ng</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> di<sub>rec</sub>ti<sub>ons,</sub> <sub>w</sub>hi<sub>c</sub>h dil<sub>u</sub>t<sub>e</sub> th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>s</sub>i<sub>gna</sub>l<sub>.</sub> C<sub>onverse</sub>l<sub>y,</sub> PSL-RGHA d<sub>e</sub>m<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>t<sub>es</sub> r<sub>e</sub>m<sub>a</sub>rk<sub>a</sub>bl<sub>e</sub> r<sub>o</sub>b<sub>us</sub>tn<sub>ess, co</sub>nfirmin<sub>g</sub> th<sub>a</sub>t th<sub>e</sub> RGPC <sub>mo</sub>d<sub>u</sub>l<sub>e e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> filt<sub>ers ou</sub>t th<sub>e</sub> t<sub>a</sub>il <sub>no</sub>i<sub>se</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d b<sub>y a</sub>dditi<sub>ona</sub>l <sub>c</sub>li<sub>en</sub>t<sub>s, ensur</sub>i<sub>ng</sub> th<sub>a</sub>t th<sub>e re</sub>f<sub>erence</sub> di<sub>rec</sub>ti<sub>on rema</sub>i<sub>ns</sub> <sub>ro</sub>b<sub>us</sub>t <sub>regar</sub>dl<sub>ess o</sub>f th<sub>e sys</sub>t<sub>em sca</sub>l<sub>e.</sub>

5.3.3 Impact of Data Heterogeneity. Fig. 11 evaluates the model robustness a<sub>g</sub>ainst statistical hetero<sub>g</sub>eneit<sub>y</sub>. As the hetero<sub>g</sub>eneit<sub>y</sub> increase (� decreases from 0.7 to 0.1), all baseline methods sufer f<sub>rom per</sub>f<sub>ormance</sub> d<sub>egra</sub>d<sub>a</sub>ti<sub>on.</sub> Thi<sub>s con</sub>fi<sub>rms</sub> th<sub>a</sub>t <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>aggre-</sub> <sub>g</sub>ation mechanisms are hi<sub>g</sub>hl<sub>y</sub> sensitive to the <sub>g</sub>radient bias induced b<sub>y s</sub>k<sub>ewe</sub>d d<sub>a</sub>t<sub>a</sub> di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n<sub>s.</sub> In <sub>co</sub>ntr<sub>as</sub>t<sub>,</sub> PSL-RGHA <sub>e</sub>xhibit<sub>s e</sub>x-<sub>cep</sub>ti<sub>o</sub>n<sub>a</sub>l <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l r<sub>es</sub>ili<sub>e</sub>n<sub>ce.</sub> E<sub>ve</sub>n <sub>u</sub>nd<sub>e</sub>r <sub>e</sub>xtr<sub>e</sub>m<sub>e</sub> n<sub>o</sub>n-IID <sub>se</sub>ttin<sub>gs</sub> (� = 0.1), it maintains su<sub>p</sub>erior accurac<sub>y</sub> with onl<sub>y</sub> mar<sub>g</sub>inal de<sub>g</sub>rad<sub>a</sub>ti<sub>o</sub>n <sub>co</sub>m<sub>pa</sub>r<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> IID <sub>sce</sub>n<sub>a</sub>ri<sub>os.</sub> Thi<sub>s</sub> r<sub>o</sub>b<sub>us</sub>tn<sub>ess va</sub>lid<sub>a</sub>t<sub>es</sub> th<sub>a</sub>t th<sub>e propose</sub>d GHA <sub>mec</sub>h<sub>an</sub>i<sub>sm e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> d<sub>ecoup</sub>l<sub>es</sub> l<sub>oca</sub>l d<sub>a</sub>t<sub>a</sub> bi<sub>as</sub> f<sub>rom</sub> th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> di<sub>rec</sub>ti<sub>on,</sub> <sub>ena</sub>bli<sub>ng</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l t<sub>o</sub> <sub>a</sub>d<sub>ap</sub>t t<sub>o severe</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on s</sub>hift<sub>s.</sub>

5.3.4 Ablation Study. We first isolate the contribution of the RGPC module b<sub>y</sub> com<sub>p</sub>arin<sub>g</sub> it a<sub>g</sub>ainst two variants: (i) Non-RGPC (Naive Strate<sub>gy</sub>), which omits the robust construction ste<sub>p</sub>s, and (ii) Rand-RGPC (Stochastic Strate<sub>gy</sub>), which constructs the reference from <sub>a</sub> r<sub>a</sub>nd<sub>o</sub>m <sub>su</sub>b<sub>se</sub>t <sub>o</sub>f <sub>c</sub>li<sub>e</sub>nt <sub>g</sub>r<sub>a</sub>di<sub>e</sub>nt<sub>s.</sub> A<sub>s</sub> <sub>s</sub>h<sub>ow</sub>n in Fi<sub>g.</sub> 12<sub>,</sub> RGPC consistentl<sub>y</sub> out<sub>p</sub>erforms both variants. The su<sub>p</sub>eriorit<sub>y</sub> over Rand-RGPC reveals that reference <sub>q</sub>ualit<sub>y</sub> is critical: rel<sub>y</sub>in<sub>g</sub> on random <sub>su</sub>b<sub>se</sub>t<sub>s</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uces s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c no</sub>i<sub>se</sub> th<sub>a</sub>t d<sub>es</sub>t<sub>a</sub>bili<sub>zes</sub> th<sub>e a</sub>li<sub>gnmen</sub>t anchor. Similarl<sub>y,</sub> the <sub>g</sub>a<sub>p</sub> over Non-RGPC confirms that witho<sub>u</sub>t the W<sub>e</sub>i<sub>sz</sub>f<sub>e</sub>ld<sub>-</sub>b<sub>ase</sub>d filt<sub>er</sub>i<sub>ng,</sub> th<sub>e re</sub>f<sub>erence</sub> di<sub>rec</sub>ti<sub>on</sub> i<sub>s eas</sub>il<sub>y s</sub>k<sub>ewe</sub>d b<sub>y</sub> <sub>ou</sub>tli<sub>e</sub>r<sub>s</sub> in n<sub>o</sub>n-IID d<sub>a</sub>t<sub>a.</sub> RGPC <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y p</sub>r<sub>ov</sub>id<sub>es a s</sub>t<sub>a</sub>bl<sub>e a</sub>n<sub>c</sub>h<sub>o</sub>r f<sub>or</sub> th<sub>e su</sub>b<sub>sequen</sub>t <sub>a</sub>li<sub>gnmen</sub>t<sub>.</sub>

![](images/775b79e9e2a9d04fbd298f32af112a99d7027cf6359273581b7810925b4f6c83.jpg)  
(a) CIFAR-10(IID).

![](images/3676ab8354cb6e721e086208f2f58cccbf431c6ced2935df69c61a214da1e86e.jpg)  
(b) CIFAR-100(IID).

![](images/1a10542184359d8c860f9a651e7583457920d8f2a4764b94d41d49a4be4fe83c.jpg)  
(c) FMNIST(IID). Round

![](images/651dfc295775a589df6685f2f7b2e5f908e999dd0a7277d0ed34474a5d479d0a.jpg)  
<sup>RoundRound Round</sup>(d) CIFAR-10(non-IID).<sup>Round</sup>

![](images/bd2da2a0dbb3c3babdca4c223926fd3c4ecb792e8f539a55b858c883aea38adf.jpg)

![](images/7f5f770f7b90952b3d49f201a2479d58e297271f50d087c5ff631bf40d4f8771.jpg)  
<sup>RouRound</sup>(e) CIFAR-100(non-IID).  
<sup>RRound RRound</sup>(f) FMNIST(non-IID).

Figure 12: Ablation study of RGPC on CIFAR-10, CIFAR-100,  
and FMNIST under IID and non-IID settings (ResNet-18).90 60 9590 60 95  
![](images/be2f1177e72d5bbebf8346560c983ea40494354a2d6f5d4a3cbbb25f25c049b8.jpg)

![](images/197c330f98a6afc5f3f382015f31e85c7006f333173791cf6210167097f56e69.jpg)  
<sup>Round</sup>(a) CIFAR-10 (IID).<sup>Round</sup>

![](images/0fb595a72bd8bcaebd280789281729b48fdb901acffd037f2d0924b7a97650ea.jpg)

![](images/3491af5eded7f58f3210cfd5f3a7a7d9ffd8fe6684336098f3fb368726d54d4d.jpg)

<sup>RoundRound</sup>(b) CIFAR-100 (IID).  
![](images/153b2a95720a8717509aa4133c0eec873f0b3529664602786f30f80910bee1a1.jpg)  
<sup>0</sup> <sup>100</sup> <sup>200</sup><sub>Round</sub><sup>0</sup> <sup>100</sup> <sup>200</sup><sub>Round</sub> (e) CIFAR-100(non-IID).

(c) FMNIST (IID).  
![](images/00dd7687751ee06eb521ba4c922f7d20889405233e48c29a6a981f1ceb4b4f89.jpg)  
<sup>0</sup> <sup>100</sup> <sup>200</sup><sub>Round</sub><sup>0</sup> <sup>100</sup> <sup>200</sup><sub>Round</sub> <sup>RoundRound</sup>(f) FMNIST(non-IID).  
Figure 13: Ablation study of GHA on CIFAR-10, CIFAR-100, and FMNIST under IID and non-IID settings (ResNet-18).

We then <sub>v</sub>erif<sub>y</sub> the necessit<sub>y</sub> of the GHA mechanism b<sub>y</sub> examinin<sub>g</sub> the <sub>p</sub>erformance <sub>g</sub>a<sub>p</sub> amon<sub>g</sub> f<sub>u</sub>ll PSL-RGHA framework<sub>,</sub> baseline PSL (without ali<sub>g</sub>nment) and SFL (naive a<sub>gg</sub>re<sub>g</sub>ation), as ill<sub>u</sub>strated in Fi<sub>g</sub>. 13. Ex<sub>p</sub>erimental res<sub>u</sub>lts demonstrate that GHA <sub>y</sub>ields substantial accurac<sub>y</sub> <sub>g</sub>ains<sub>,</sub> confirmin<sub>g</sub> that sim<sub>p</sub>l<sub>y</sub> a<sub>gg</sub>re<sub>g</sub>atin<sub>g g</sub>radients (SFL) or allowin<sub>g</sub> inde<sub>p</sub>endent u<sub>p</sub>dates (PSL) is insufi-<sub>c</sub>i<sub>e</sub>nt <sub>u</sub>nd<sub>e</sub>r h<sub>e</sub>t<sub>e</sub>r<sub>oge</sub>n<sub>e</sub>it<sub>y.</sub> Th<sub>e e</sub>f<sub>ec</sub>ti<sub>ve</sub>n<sub>ess o</sub>f GHA <sub>s</sub>t<sub>e</sub>m<sub>s</sub> fr<sub>o</sub>m it<sub>s</sub> a<sup>b</sup>i<sup>l</sup>ity to active<sup>l</sup>y recti<sup>f</sup>y gra<sup>d</sup>ient con<sup>fl</sup>icts. By projecting con<sup>fl</sup>icting <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>on</sub>t<sub>o</sub> th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> h<sub>a</sub>lf<sub>-space,</sub> GHA <sub>e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>es</sub> d<sub>es</sub>t<sub>ruc</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>er</sub>f<sub>erence</sub> <sub>w</sub>hil<sub>e</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>ve</sub> l<sub>oca</sub>l f<sub>ea</sub>t<sub>ures,</sub> th<sub>ere</sub>b<sub>y</sub> convertin<sub>g</sub> o<sub>p</sub>timization drift into stable conver<sub>g</sub>ence.

## 6 Conclusion

In this <sub>p</sub>a<sub>p</sub>er<sub>,</sub> we <sub>p</sub>ro<sub>p</sub>osed PSL-RGHA to resolve the f<sub>u</sub>ndamental conflict between client-side <sub>p</sub>arallelism and trainin<sub>g</sub> stabilit<sub>y</sub> in SL. Unlike traditional a<sub>pp</sub>roaches that rel<sub>y</sub> on naive model avera<sub>g</sub>in<sub>g,</sub> PSL-RGHA intr<sub>o</sub>d<sub>uces a geo</sub>m<sub>e</sub>tri<sub>c pe</sub>r<sub>spec</sub>ti<sub>ve</sub> t<sub>o</sub> di<sub>s</sub>trib<sub>u</sub>t<sub>e</sub>d <sub>op</sub>timiz<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> Thr<sub>oug</sub>h RGPC <sub>a</sub>nd GHA<sub>,</sub> th<sub>e</sub> fr<sub>a</sub>m<sub>ewo</sub>rk <sub>ac</sub>ti<sub>ve</sub>l<sub>y</sub> filt<sub>e</sub>r<sub>s</sub> out directional noise and rectifies diver<sub>g</sub>ent u<sub>p</sub>dates<sub>,</sub> decou<sub>p</sub>lin<sub>g</sub> <sup>l</sup>oca<sup>l</sup> <sup>d</sup>ata <sup>b</sup>ias <sup>f</sup>rom t<sup>h</sup>e g<sup>l</sup>o<sup>b</sup>a<sup>l</sup> optimization trajectory. Extensive ex-<sub>p</sub>eriments <sub>v</sub>alidate that PSL-RGHA achie<sub>v</sub>es s<sub>up</sub>erior <sub>p</sub>erformance<sub>,</sub> <sub>p</sub>rovin<sub>g</sub> that ex<sub>p</sub>licit <sub>g</sub>radient ali<sub>g</sub>nment is crucial for robust learnin<sub>g</sub> on non-IID data. These findin<sub>g</sub>s o<sub>p</sub>en new avenues for desi<sub>g</sub>nin<sub>g</sub> a<sub>gg</sub>re<sub>g</sub>ation-free distributed learnin<sub>g</sub> s<sub>y</sub>stems. Movin<sub>g</sub> forward<sub>,</sub> we aim to extend this <sub>g</sub>eometric ali<sub>g</sub>nment <sub>p</sub>rinci<sub>p</sub>le to as<sub>y</sub>nchronous t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>scenar</sub>i<sub>os</sub> <sub>an</sub>d l<sub>arger-sca</sub>l<sub>e</sub> f<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l fi<sub>ne-</sub>t<sub>un</sub>i<sub>ng.</sub>

## Acknowledgments

T<sup>h</sup>is wor<sup>k</sup> was supporte<sup>d</sup> <sup>b</sup>y t<sup>h</sup>e Fujian Provincia<sup>l</sup> Science an<sup>d</sup> Tec<sup>h</sup>no<sup>l</sup>ogy P<sup>l</sup>an Project un<sup>d</sup>er Grant No. 2024H6020, an<sup>d</sup> t<sup>h</sup>e Green an<sup>d</sup> Inte<sup>ll</sup>igent S<sup>h</sup>ip Project in Fujian Province un<sup>d</sup>er Grant No. CBG4N21-4-4.

## References

[1] Durmus Al<sub>p</sub> Emre Acar, Yue Zhao, Ramon Matas Navarro, Matthew Mattina, Paul N. Whatmou<sub>g</sub>h<sub>,</sub> and Venkatesh Sali<sub>g</sub>rama. 2021. Federated Learnin<sub>g</sub> Based on Dynamic Regularization. In Proc. ICLR

[2] Miao Cao, Lishun Wan<sub>g</sub>, Huan Wan<sub>g</sub>, Guo<sub>q</sub>in<sub>g</sub> Wan<sub>g</sub>, and Xin Yuan. 2024. Towards Real-time Video Compressive Sensing on Mobile Devices. In Proc. ACM MM.

[3] Huanchen<sub>g</sub> Chen and Haris Vikalo. 2024. Hetero<sub>g</sub>eneit<sub>y</sub>-Guided Client Sam<sub>p</sub>lin<sub>g</sub>: Towards Fast and Eficient Non-IID Federated Learning. In Proc. NeurIPS.

[4] Shen<sub>g</sub>chao Chen, Guodon<sub>g</sub> Lon<sub>g</sub>, Tao Shen, Jin<sub>g</sub> Jian<sub>g</sub>, and Chen<sub>gq</sub>i Zhan<sub>g</sub>. 2024. Federated Prompt Learning for Weather Foundation Models on Devices. In Proc. IJCAI.

[5] Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, D<sub>av</sub>id L<sub>e</sub>r<sub>oy,</sub> Clém<sub>e</sub>nt D<sub>ou</sub>m<sub>ou</sub>r<sub>o,</sub> Thib<sub>au</sub>lt Gi<sub>sse</sub>lbr<sub>ec</sub>ht<sub>,</sub> Fr<sub>a</sub>n<sub>cesco</sub> C<sub>a</sub>lt<sub>ag</sub>ir<sub>o</sub>n<sub>e,</sub> Thibaut Lavril, Maël Primet, and Jose<sub>p</sub>h Dureau. 2018. Sni<sub>p</sub>s Voice Platform: an embedded S<sub>p</sub>oken Lan<sub>g</sub>ua<sub>g</sub>e Understandin<sub>g</sub> s<sub>y</sub>stem for <sub>p</sub>rivate-b<sub>y</sub>-desi<sub>g</sub>n voice interfaces. arXiv preprint arXiv: 1805.10190 (2018).

[6] Don<sub>g</sub>shan<sub>g</sub> Den<sub>g</sub>, Xuan<sub>g</sub>ou Wu, Tao Zhan<sub>g</sub>, Xian<sub>gy</sub>un Tan<sub>g</sub>, Hon<sub>gy</sub>an<sub>g</sub> Du, Jiawen Kan<sub>g</sub>, Ji<sub>q</sub>ian<sub>g</sub> Liu, and Dusit Ni<sub>y</sub>ato. 2024. FedASA: A Personalized Federated Learnin<sub>g</sub> With Ada<sub>p</sub>tive Model A<sub>gg</sub>re<sub>g</sub>ation for Hetero<sub>g</sub>eneous Mobile Edge Computing. IEEE Trans. Mob. Comput. 23, 12 (2024), 14787–14802.

[7] Zihan Fan<sub>g</sub>, Zhen<sub>g</sub> Lin, Senkan<sub>g</sub> Hu, Yanan Ma, Yihan<sub>g</sub> Tao, Yi<sub>q</sub>in Den<sub>g</sub>, Xianhao Ch<sub>e</sub>n<sub>,</sub> <sub>a</sub>nd Y<sub>ugua</sub>n<sub>g</sub> F<sub>a</sub>n<sub>g.</sub> 2026<sub>.</sub> HF<sub>e</sub>dM<sub>o</sub>E: R<sub>esou</sub>r<sub>ce</sub>-<sub>awa</sub>r<sub>e</sub> H<sub>e</sub>t<sub>e</sub>r<sub>oge</sub>n<sub>eous</sub> Federated Learning with Mixture-of-Experts. arXiv preprint arXiv:2601.00583 (2026).

[8] Zihan Fan<sub>g</sub>, Zhen<sub>g</sub> Lin, Senkan<sub>g</sub> Hu, Yihan<sub>g</sub> Tao, Yi<sub>q</sub>in Den<sub>g</sub>, Xianhao Chen, and Yu<sub>g</sub>uan<sub>g</sub> Fan<sub>g</sub>. 2025. D<sub>y</sub>namic uncertaint<sub>y</sub>-aware multimodal fusion for outdoor health monitoring. arXiv preprint arXiv:2508.09085 (2025).

[9] Zihan Fan<sub>g</sub>, Qianru Wan<sub>g</sub>, Haonan An, Zhen<sub>g</sub> Lin, Yi<sub>q</sub>in Den<sub>g</sub>, Xianhao Chen, and Yu<sub>g</sub>uan<sub>g</sub> Fan<sub>g</sub>. 2026. A<sub>gg</sub>re<sub>g</sub>ation Ali<sub>g</sub>nment for Federated Learnin<sub>g</sub> with Mixture-of-Experts under Data Heterogeneity. arXiv preprint arXiv:2603.21276 (2026).

[10] Zhen Fan<sub>g</sub>, Miao Yan<sub>g</sub>, Zehan<sub>g</sub> Lin, Zhen<sub>g</sub> Lin, Zihan Fan<sub>g</sub>, Zon<sub>gy</sub>uan Zhan<sub>g</sub>, Ti<sub>a</sub>n<sub>ya</sub>n<sub>g</sub> D<sub>ua</sub>n<sub>,</sub> D<sub>o</sub>n<sub>g</sub> H<sub>ua</sub>n<sub>g, a</sub>nd Sh<sub>u</sub>nzhi Zh<sub>u.</sub> 2026<sub>.</sub> NSC-SL: A B<sub>a</sub>nd<sub>w</sub>idth-A<sub>wa</sub>r<sub>e</sub> N<sub>eu</sub>r<sub>a</sub>l S<sub>u</sub>b<sub>space</sub> C<sub>o</sub>m<sub>p</sub>r<sub>ess</sub>i<sub>o</sub>n f<sub>o</sub>r C<sub>o</sub>mm<sub>u</sub>ni<sub>ca</sub>ti<sub>o</sub>n-Efi<sub>c</sub>i<sub>e</sub>nt S<sub>p</sub>lit L<sub>ea</sub>rnin<sub>g.</sub> In Proc. ICASSP. 2156–2160.

[11] Lian<sub>g</sub> Gao, Huazhu Fu, Li Li, Yin<sub>g</sub>wen Chen, Min<sub>g</sub> Xu, and Chen<sub>g</sub>-Zhon<sub>g</sub> Xu. 2022<sub>.</sub> F<sub>e</sub>dDC: F<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d L<sub>ea</sub>rnin<sub>g</sub> <sub>w</sub>ith N<sub>o</sub>n-IID D<sub>a</sub>t<sub>a</sub> <sub>v</sub>i<sub>a</sub> L<sub>oca</sub>l Drift D<sub>ecoup</sub>lin<sub>g</sub> and Correction. In Proc. CVPR.

[12] Yamin<sub>g</sub> Guo, Kai Guo, Xiaofen<sub>g</sub> Cao, Tieru Wu, and Yi Chan<sub>g</sub>. 2023. Out-of-Di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f F<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d L<sub>ea</sub>rnin<sub>g v</sub>i<sub>a</sub> Im<sub>p</sub>li<sub>c</sub>it In<sub>va</sub>ri<sub>a</sub>nt R<sub>e</sub>l<sub>a</sub> tionships. In Proc. ICML.

[13] Kaimin<sub>g</sub> He, Xian<sub>gy</sub>u Zhan<sub>g</sub>, Shao<sub>q</sub>in<sub>g</sub> Ren, and Jian Sun. 2016. Dee<sub>p</sub> Residual Learning for Image Recognition. In Proc. CVPR.

[14] Kaili Jin, Li Xu, Xiaodin<sub>g</sub> Wan<sub>g</sub>, Sun-Yuan Hsieh, Jie Wu, and Limei Lin. 2025. Fed CPD: P<sub>e</sub>r<sub>so</sub>n<sub>a</sub>liz<sub>e</sub>d F<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d L<sub>ea</sub>rnin<sub>g w</sub>ith Pr<sub>o</sub>t<sub>o</sub>t<sub>ype</sub>-Enh<sub>a</sub>n<sub>ce</sub>d R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n and Memory Distillation. In Proc. IJCAI.

[15] Praveen Joshi, Chandra Tha<sub>p</sub>a, Se<sub>y</sub>it Camte<sub>p</sub>e, Mohammed Hasanuzzaman, Ted Sc<sub>u</sub>ll<sub>y,</sub> and Haithem Afli. 2021. S<sub>p</sub>litfed Learnin<sub>g w</sub>itho<sub>u</sub>t Client-side S<sub>y</sub>nchroniza ti<sub>o</sub>n: An<sub>a</sub>l zin Cli<sub>e</sub>nt-<sub>s</sub>id<sub>e</sub> S lit N<sub>e</sub>t<sub>wo</sub>rk P<sub>o</sub>rti<sub>o</sub>n Siz<sub>e</sub> t<sub>o</sub> O<sub>ve</sub>r<sub>a</sub>ll P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce.</sub> arXiv preprint arXiv:2109.09246 (2021).

[16] Sai Praneeth Karimiredd<sub>y</sub>, Sat<sub>y</sub>en Kale, Mehr<sub>y</sub>ar Mohri, Sashank J. Reddi, Seb<sub>as</sub>ti<sub>a</sub>n U<sub>.</sub> Sti<sub>c</sub>h<sub>, a</sub>nd An<sub>a</sub>nd<sub>a</sub> Th<sub>ee</sub>rth<sub>a</sub> S<sub>u</sub>r<sub>es</sub>h<sub>.</sub> 2020<sub>.</sub> SCAFFOLD: St<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> Controlled Averaging for Federated Learning. In Proc. ICML.

[17] Minsu Kim, Alexander C. DeRieux, and Walid Saad. 2022. A Bar<sub>g</sub>ainin<sub>g</sub> Game for Personalized, Energy Eficient Split Learning over Wireless Networks. arXiv preprint arXiv:2212.06107 (2022).

[18] Minsu Kim, Walid Saad, Mérouane Debbah, and Choon<sub>g</sub> Seon Hon<sub>g</sub>. 2024. S<sub>p</sub>aFL: C<sub>o</sub>mm<sub>u</sub>ni<sub>ca</sub>ti<sub>o</sub>n-Efi<sub>c</sub>i<sub>e</sub>nt F<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d L<sub>ea</sub>rnin<sub>g</sub> With S<sub>pa</sub>r<sub>se</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s</sub> And L<sub>ow</sub> Computational Overhead. In Proc. NeurIPS.

[19] Alex Krizhevsky. 2009. Learning Multiple Layers of Features from Tiny Images. T<sub>ec</sub>hni<sub>ca</sub>l R<sub>epo</sub>rt<sub>.</sub> Uni<sub>ve</sub>r<sub>s</sub>it<sub>y o</sub>f T<sub>o</sub>r<sub>o</sub>nt<sub>o.</sub>

[20] Yann LeCun, Léon Bottou, Yoshua Ben<sub>g</sub>io, and Patrick Hafner. 1998. Gradientbased learning applied to document recognition. Proc. IEEE 86, 11 (1998), 2278– 2324.

[21] Ro<sub>y</sub>son Lee, Min<sub>y</sub>oun<sub>g</sub> Kim, Da Li, Xinchi Qiu, Timoth<sub>y</sub> M. Hos<sub>p</sub>edales, Ferenc H<sub>us</sub>z<sub>a</sub>r<sub>,</sub> <sub>a</sub>nd Ni<sub>c</sub>h<sub>o</sub>l<sub>as</sub> D<sub>.</sub> L<sub>a</sub>n<sub>e.</sub> 2023<sub>.</sub> F<sub>e</sub>dL2P: F<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d L<sub>ea</sub>rnin<sub>g</sub> t<sub>o</sub> P<sub>e</sub>r<sub>so</sub>n<sub>a</sub>liz<sub>e.</sub> In Proc. NeurIPS.

[22] Junnan Li, Don<sub>g</sub>xu Li, Silvio Savarese, and Steven C. H. Hoi. 2023. BLIP-2: B<sub>oo</sub>t<sub>s</sub>tr<sub>a</sub> in L<sub>a</sub>n <sub>ua e</sub>-Im<sub>a e</sub> Pr<sub>e</sub>-tr<sub>a</sub>inin <sub>w</sub>ith Fr<sub>o</sub>z<sub>e</sub>n Im<sub>a e</sub> En<sub>co</sub>d<sub>e</sub>r<sub>s a</sub>nd Large Language Models. In Proc. ICML.

[23] Qinbin Li, Bin<sub>g</sub>shen<sub>g</sub> He, and Dawn Son<sub>g</sub>. 2021. Model-Contrastive Federated Learning. In Proc. CVPR.

[24] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Vir<sub>g</sub>ini<sub>a</sub> Smith<sub>.</sub> 2020<sub>.</sub> F<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d O<sub>p</sub>timiz<sub>a</sub>ti<sub>o</sub>n in H<sub>e</sub>t<sub>e</sub>r<sub>oge</sub>n<sub>eous</sub> N<sub>e</sub>t<sub>wo</sub>rk<sub>s.</sub> In Proc. MLSys.

[25] Zijian Li, Zehong Lin, Jiawei Shao, Yuyi Mao, and Jun Zhang. 2024. FedCiR: Client-Invariant Representation Learning for Federated Non-IID Features. IEEE Trans. Mob. Comput. 23, 11 (2024), 10509–10522.

[26] Zhen<sub>g</sub> Lin, Zhe Chen, Xianhao Chen, Wei Ni, and Yue Gao. 2026. HASFL: Hetero<sub>g</sub>eneit<sub>y</sub>-aware S<sub>p</sub>lit Federated Learnin<sub>g</sub> over Ed<sub>g</sub>e Com<sub>p</sub>utin<sub>g</sub> S<sub>y</sub>stems. IEEE Trans. Mobile Comput. (2026).

[27] Zhen<sub>g</sub> Lin, Zhe Chen, Zihan Fan<sub>g</sub>, Xianhao Chen, Xion<sub>g</sub> Wan<sub>g</sub>, and Yue Gao. 2024<sub>.</sub> F<sub>e</sub>dSN: A f<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d l<sub>ea</sub>rnin<sub>g</sub> fr<sub>a</sub>m<sub>ewo</sub>rk <sub>ove</sub>r h<sub>e</sub>t<sub>e</sub>r<sub>oge</sub>n<sub>eous</sub> LEO <sub>sa</sub>t<sub>e</sub>llit<sub>e</sub> networks. IEEE Transactions on Mobile Computing 24, 3 (2024), 1293–1307.

[28] Zheng Lin, Guanqiao Qu, Qiyuan Chen, Xianhao Chen, Zhe Chen, and Kaibin Huan<sub>g</sub>. 2025. Pushin<sub>g</sub> Lar<sub>g</sub>e Lan<sub>g</sub>ua<sub>g</sub>e Models to the 6G Ed<sub>g</sub>e: Vision<sub>,</sub> Challen<sub>g</sub>es<sub>,</sub> and Opportunities. IEEE Communication Magazine (2025).

[29] Zhen<sub>g</sub> Lin, Guan<sub>q</sub>iao Qu, Wei Wei, Xianhao Chen, and Kin K Leun<sub>g</sub>. 2025. Ada<sub>p</sub>tsfl: Ada<sub>p</sub>ti<sub>v</sub>e S<sub>p</sub>lit Federated Learnin<sub>g</sub> in Reso<sub>u</sub>rce-Constrained Ed<sub>g</sub>e Net works. IEEE Trans. Netw. (2025).

[30] Zhen<sub>g</sub> Lin, Zhanwei Wan<sub>g</sub>, Yuxin Zhan<sub>g</sub>, Yafei Wan<sub>g</sub>, Ons Aouedi, Zihan Fan<sub>g</sub>, Jinfen<sub>g</sub> Xu, Jin<sub>g</sub> Yan<sub>g</sub>, Vu N<sub>g</sub>u<sub>y</sub>en Ha, Yuanwei Liu, Mohamed-Slim Alouini, and S<sub>y</sub>m<sub>eo</sub>n Ch<sub>a</sub>tzin<sub>o</sub>t<sub>as.</sub> 2026<sub>.</sub> S<sub>a</sub>t<sub>e</sub>llit<sub>e</sub> Int<sub>e</sub>lli<sub>ge</sub>n<sub>ce</sub>: Di<sub>s</sub>trib<sub>u</sub>t<sub>e</sub>d Tr<sub>a</sub>inin<sub>g,</sub> C<sub>o</sub>ll<sub>a</sub>b orative Inference, and LLM Adaptation in LEO Constellation. arXiv preprint arXiv:2610.35282 (2026).

[31] Zhen<sub>g</sub> Lin, Yuxin Zhan<sub>g</sub>, Zhe Chen, Zihan Fan<sub>g</sub>, Xianhao Chen, Praneeth Ve<sub>p</sub>akomma, Wei Ni, Jun Luo, and Yue Gao. 2025. HS<sub>p</sub>litLoRA: A Hetero<sub>g</sub>en<sub>eous</sub> S<sub>p</sub>lit P<sub>a</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r-Efi<sub>c</sub>i<sub>e</sub>nt Fin<sub>e</sub>-T<sub>u</sub>nin<sub>g</sub> Fr<sub>a</sub>m<sub>ewo</sub>rk f<sub>o</sub>r L<sub>a</sub>r<sub>ge</sub> L<sub>a</sub>n<sub>guage</sub> Models. arXiv preprint arXiv:2505.02795 (2025).

[32] Zhen<sub>g</sub> Lin, Guan<sub>gy</sub>u Zhu, Yi<sub>q</sub>in Den<sub>g</sub>, Xianhao Chen, Yue Gao, Kaibin Huan<sub>g</sub>, and Y<sub>ugua</sub>n<sub>g</sub> F<sub>a</sub>n<sub>g.</sub> 2024<sub>.</sub> Efi<sub>c</sub>i<sub>e</sub>nt P<sub>a</sub>r<sub>a</sub>ll<sub>e</sub>l S<sub>p</sub>lit L<sub>ea</sub>rnin<sub>g</sub> O<sub>ve</sub>r R<sub>esou</sub>r<sub>ce</sub>-C<sub>o</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d Wireless Edge Networks. IEEE Trans. Mob. Comput. 23, 10 (2024), 9224–9239.

[33] Yi Liu, Con<sub>g</sub> Wan<sub>g</sub>, and Xin<sub>g</sub>lian<sub>g</sub> Yuan. 2025. FedMobile: Enablin<sub>g</sub> Knowled<sub>g</sub>e Contrib<sub>u</sub>tion-a<sub>w</sub>are M<sub>u</sub>lti-modal Federated Learnin<sub>g</sub> <sub>w</sub>ith Incom<sub>p</sub>lete Modalities. In Proc. WWW.

[34] Zechun Liu, Chan<sub>g</sub>shen<sub>g</sub> Zhao, Forrest N. Iandola, Chen Lai, Yuandon<sub>g</sub> Tian, I<sub>g</sub>or Fedorov<sub>,</sub> Y<sub>u</sub>n<sub>y</sub>an<sub>g</sub> Xion<sub>g,</sub> Ernie Chan<sub>g,</sub> Yan<sub>gy</sub>an<sub>g</sub> Shi<sub>,</sub> Ra<sub>g</sub>h<sub>u</sub>raman Krishn<sub>a</sub>m<sub>oo</sub>rthi<sub>,</sub> Li<sub>a</sub>n<sub>g</sub>zh<sub>e</sub>n L<sub>a</sub>i<sub>,</sub> <sub>a</sub>nd Vik<sub>as</sub> Ch<sub>a</sub>ndr<sub>a.</sub> 2024<sub>.</sub> M<sub>o</sub>bil<sub>e</sub>LLM: O<sub>p</sub>timizin<sub>g</sub> Sub-billion Parameter Language Models for On-Device Use Cases. In Proc. ICML.

[35] Jinbo Pen<sub>g</sub>, Zhe Chen, Zhen<sub>g</sub> Lin, Haoxuan Yuan, Zihan Fan<sub>g</sub>, Lin<sub>g</sub>zhon<sub>g</sub> Bao, Zihan<sub>g</sub> Son<sub>g</sub>, Yin<sub>g</sub> Li, Jin<sub>g</sub> Ren, and Yue Gao. 2024. SUMS: Snifin<sub>g</sub> Unknown Multiband Signals under Low Sampling Rates. IEEE Trans. Mobile Comput. (2024).

[36] Youban<sub>g</sub> Sun, Zitao Li, Yalian<sub>g</sub> Li, and Bolin Din<sub>g</sub>. 2024. Im<sub>p</sub>rovin<sub>g</sub> LoRA in Privacy-preserving Federated Learning. In Proc. ICLR.

[37] Yuxia Sun, Aoxian<sub>g</sub> Sun, Si<sub>y</sub>i Pan, Zhixiao Fu, and Jin<sub>g</sub>cai Guo. 2025. FedAPA: Server-side Gradient-Based Ada<sub>p</sub>tive Personalized A<sub>gg</sub>re<sub>g</sub>ation for Federated Learning on Heterogeneous Data. In Proc. IJCAI.

[38] Petroc Ta<sub>y</sub>lor. 2025. Data <sub>g</sub>eneration volume worldwide 2010-2029. htt<sub>p</sub>s://www. statista.com/statistics/871513/worldwide-data-created/. Accessed: 2025-12-05.

[39] Chandra Tha<sub>p</sub>a, Mahawa<sub>g</sub>a Arachchi<sub>g</sub>e Pathum Chamikara, Se<sub>y</sub>it Camte<sub>p</sub>e, and Lichao S<sub>u</sub>n. 2022. S<sub>p</sub>litFed: When Federated Learnin<sub>g</sub> Meets S<sub>p</sub>lit Learnin<sub>g</sub>. In Proc. AAAI.

[40] Praneeth Ve<sub>p</sub>akomma, Otkrist Gu<sub>p</sub>ta, Tristan Swedish, and Ramesh Raskar. 2018. S<sub>p</sub>lit L<sub>ea</sub>rnin<sub>g</sub> f<sub>o</sub>r H<sub>ea</sub>lth: Di<sub>s</sub>trib<sub>u</sub>t<sub>e</sub>d D<sub>eep</sub> L<sub>ea</sub>rnin<sub>g w</sub>ith<sub>ou</sub>t Sh<sub>a</sub>rin<sub>g</sub> R<sub>aw</sub> P<sub>a</sub>ti<sub>e</sub>n Data. arXiv preprint arXiv:1812.00564 (2018).

[41] Haozhao Wan<sub>g</sub>, Wenchao Xu, Yunfen<sub>g</sub> Fan, Ruixuan Li, and Pan Zhou. 2023. AOCC-FL: F<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d L<sub>ea</sub>rnin<sub>g w</sub>ith Ali<sub>g</sub>n<sub>e</sub>d O<sub>ve</sub>rl<sub>app</sub>in<sub>g v</sub>i<sub>a</sub> C<sub>a</sub>libr<sub>a</sub>t<sub>e</sub>d C<sub>o</sub>mpensation. In Proc. IEEE INFOCOM.

[42] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H. Vincent Poor. 2020. Tac<sup>kl</sup>in t<sup>h</sup>e O<sup>b</sup>jective Inconsistenc Pro<sup>bl</sup>em in Hetero eneous Fe<sup>d</sup>erate<sup>d</sup> O ti mization. In Proc. NeurIPS

[43] Xin Wan<sub>g</sub>, Wenwu Zhu, Yon<sub>g</sub>hon<sub>g</sub> Tian, and Wen Gao. 2020. Multimedia Intelli gence: When Multimedia Meets Artificial Intelligence. In Proc. ACM MM.

[44] Pete Warden. 2018. S<sub>p</sub>eech Commands: A Dataset for Limited-Vocabular<sub>y</sub> S<sub>p</sub>eech Reco nition. arXiv preprint arXiv: 1804.03209 (2018).

[45] Wei Wei, Zhen<sub>g</sub> Lin, Tao Li, Xuanhen<sub>g</sub> Li, and Xianhao Chen. 2026. Pi<sub>p</sub>elinin<sub>g</sub> Split Learning in Multi-hop Edge Networks. IEEE Trans. Mob. Comput. (2026), 1–14.

[46] E. WEISZFELD. 1937. Sur le <sub>p</sub>oint <sub>p</sub>our le<sub>q</sub>uel la somme des distances de n points donnés est minimum. Tohoku Mathematical Journal, First Series 43 (1937), 355–386.

[47] Endre Weiszfeld and Frank Plastria. 2009. On the <sub>p</sub>oint for which the sum of the distances to n given points is minimum. Ann. Oper. Res. 167, 1 (2009), 7–41.

[48] Herbert Woisetschlä<sub>g</sub>er, Alexander Erben, Shi<sub>q</sub>ian<sub>g</sub> Wan<sub>g</sub>, Ruben Ma<sub>y</sub>er, and Hans-Arno Jacobsen. 2024. A Surve<sub>y</sub> on Eficient Federated Learnin<sub>g</sub> Methods

for Foundation Model Training. In Proc. IJCAI.

[49] Di Wu, Jun Bai, Yiliao Song, Junjun Chen, Wei Zhou, Yong Xiang, and Atul Sajjan<sup>h</sup>ar. 2024. Fe<sup>d</sup>Inverse: Eva<sup>l</sup>uating Privacy Lea<sup>k</sup>age in Fe<sup>d</sup>erate<sup>d</sup> Learning. In Proc. ICLR.

[50] Han Xiao, Kashif Rasul, and Roland Voll<sub>g</sub>raf. 2017. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv preprint arXiv:1708.07747 (2017).

[51] Miao Yan<sub>g</sub>, Hua Qian, Ximin Wan<sub>g</sub>, Yon<sub>g</sub> Zhou, and Hon<sub>g</sub>bin Zhu. 2022. Client Selection for Federated Learning With Label Noise. IEEE Transactions on Vehicular Technology 71, 2 (2022), 2193–2197.

[52] Miao Yan<sub>g</sub>, Ximin Wan<sub>g</sub>, Hua Qian, Yon<sub>g</sub>xin Zhu, Hon<sub>g</sub>bin Zhu, Mohsen Guizani, and Victor Chan<sub>g</sub>. 2022. An im<sub>p</sub>roved federated learnin<sub>g</sub> al<sub>g</sub>orithm for <sub>p</sub>rivac<sub>y</sub> preserving in cybertwin-driven 6G system. IEEE Trans. Ind. Inf. 18, 10 (2022), 6733–6742.

[53] Min<sub>g</sub>kun Yan<sub>g</sub>, Ran Zhu, Qin<sub>g</sub> Wan<sub>g</sub>, and Jie Yan<sub>g</sub>. 2024. FedTrans: Client-Transparent Utility Estimation for Robust Federated Learning. In Proc. ICLR.

[54] Rui Ye, Rui Ge, Xin<sub>y</sub>u Zhu, Jin<sub>gy</sub>i Chai, Yaxin Du, Yan<sub>g</sub> Liu, Yanfen<sub>g</sub> Wan<sub>g</sub>, and Sih<sub>e</sub>n<sub>g</sub> Ch<sub>e</sub>n<sub>.</sub> 2024<sub>.</sub> F<sub>e</sub>dLLM-B<sub>e</sub>n<sub>c</sub>h: R<sub>ea</sub>li<sub>s</sub>ti<sub>c</sub> B<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rk<sub>s</sub> f<sub>o</sub>r F<sub>e</sub>d<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d L<sub>ea</sub>rnin<sub>g</sub> of Large Language Models. In Proc. NeurIPS.

[55] Tianhe Yu, Saurabh Kumar, Abhishek Gu<sub>p</sub>ta, Ser<sub>g</sub>e<sub>y</sub> Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient Surgery for Multi-Task Learning. In Proc. NeurIPS.

[56] Haoxuan Yuan, Zhe Chen, Zhen<sub>g</sub> Lin, Jinbo Pen<sub>g</sub>, Zihan Fan<sub>g</sub>, Yuhan<sub>g</sub> Zhon<sub>g</sub>, Zih<sub>a</sub>n<sub>g</sub> S<sub>o</sub>n<sub>g, a</sub>nd Y<sub>ue</sub> G<sub>ao.</sub> 2025<sub>.</sub> S<sub>a</sub>tS<sub>e</sub>n<sub>se</sub>: M<sub>u</sub>lti-S<sub>a</sub>t<sub>e</sub>llit<sub>e</sub> C<sub>o</sub>ll<sub>a</sub>b<sub>o</sub>r<sub>a</sub>ti<sub>ve</sub> Fr<sub>a</sub>m<sub>ewo</sub>rk for Spectrum Sensing. IEEE Trans. Cogn. Commun. Netw. (2025).

[57] Shuan<sub>g</sub> Zen<sub>g</sub>, Pen<sub>g</sub>xin Guo, Shuai Wan<sub>g</sub>, Jianbo Wan<sub>g</sub>, Yu<sub>y</sub>in Zhou, and Liangqiong Qu. 2024. Tackling Data Heterogeneity in Federated Learning via Loss Decom osition. In Proc. MICCAI. 707–717. doi:10.1007/978-3-031-72117-5 66

[58] Hao Zhen<sub>g</sub>, Zhi<sub>g</sub>an<sub>g</sub> Hu, Liu Yan<sub>g</sub>, Mei<sub>g</sub>uan<sub>g</sub> Zhen<sub>g</sub>, Aikun Xu, and Bo<sub>y</sub>u Wan<sub>g</sub>. 2025<sub>.</sub> F<sub>e</sub>dCALM: C<sub>o</sub>nfli<sub>c</sub>t-<sub>awa</sub>r<sub>e</sub> L<sub>aye</sub>r-<sub>w</sub>i<sub>se</sub> Miti<sub>ga</sub>ti<sub>o</sub>n f<sub>o</sub>r S<sub>e</sub>l<sub>ec</sub>ti<sub>ve</sub> A<sub>gg</sub>r<sub>ega</sub>ti<sub>o</sub>n in Deeper Personalized Federated Learning. In Proc. CVPR.

[59] Weimin<sub>g</sub> Zhuan<sub>g</sub> and Lin<sub>g</sub>juan L<sub>y</sub>u. 2024. FedWon: Trium<sub>p</sub>hin<sub>g</sub> Multi-domain Federated Learning Without Normalization. In Proc. ICLR.