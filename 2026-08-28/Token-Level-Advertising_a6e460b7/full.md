# Token-Level Advertising

H<sub>a</sub>nbin<sub>g</sub> Li<sub>u</sub>   
R<sub>e</sub>nmin Uni<sub>ve</sub>r<sub>s</sub>it<sub>y o</sub>f Chin<sub>a</sub>   
Beijing, C<sup>h</sup>ina   
li<sub>u</sub>h<sub>an</sub>bi<sub>ng@ruc.e</sub>d<sub>u.cn</sub>   
B<sub>owe</sub>i Zh<sub>a</sub>n<sub>g</sub>   
R<sub>e</sub>nmin Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Chin<sub>a</sub>   
Beijing, C<sup>h</sup>ina   
2023201812@ruc.edu.cn   
Yin<sub>yu</sub> Y<sub>e</sub>   
Stanford Uni<sub>v</sub>ersit<sub>y</sub>   
St<sub>an</sub>f<sub>or</sub>d<sub>,</sub> CA<sub>,</sub> USA   
<sub>y</sub>i<sub>nyu-ye@s</sub>t<sub>an</sub>f<sub>or</sub>d<sub>.e</sub>d<sub>u</sub>   
Chan<sub>gy</sub>uan Yu   
B<sub>a</sub>id<sub>u</sub> In<sub>c.</sub>   
Beijing, C<sup>h</sup>ina   
<sub>yuc</sub>h<sub>angyuan@</sub>b<sub>a</sub>id<sub>u.com</sub>   
Qi Qi<sup>✉</sup>   
R<sub>e</sub>nmin Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f Chin<sub>a</sub>   
Beijing, C<sup>h</sup>ina   
<sub>q</sub>i.<sub>q</sub>i@ruc.edu.cn

## Abstract

Generative AI is transformin<sub>g</sub> how <sub>p</sub>eo<sub>p</sub>le access information<sub>,</sub> chall<sub>eng</sub>i<sub>ng</sub> t<sub>ra</sub>diti<sub>ona</sub>l <sub>a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng mec</sub>h<sub>an</sub>i<sub>sms</sub> b<sub>u</sub>ilt <sub>aroun</sub>d <sub>pre</sub>d<sub>e</sub>fi<sub>ne</sub>d slots. Towards <sub>g</sub>eneration-native advertisin<sub>g,</sub> we <sub>p</sub>ro<sub>p</sub>ose the Latent Advertiser Mixture Auction (LAMA), a token-level advertisin<sub>g</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> th<sub>a</sub>t <sub>em</sub>b<sub>e</sub>d<sub>s</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> i<sub>n</sub>fl<sub>uence</sub> di<sub>rec</sub>tl<sub>y</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>gen</sub> eration <sub>p</sub>rocess. Ad<sub>v</sub>ertisers re<sub>p</sub>ort local contin<sub>u</sub>ation <sub>v</sub>al<sub>u</sub>es that i<sub>n</sub>d<sub>uce a</sub>d<sub>ver</sub>ti<sub>ser-spec</sub>ifi<sub>c nex</sub>t<sub>-</sub>t<sub>o</sub>k<sub>en po</sub>li<sub>c</sub>i<sub>es,</sub> f<sub>rom w</sub>hi<sub>c</sub>h th<sub>e p</sub>l<sub>a</sub>t f<sub>orm</sub> d<sub>eco</sub>d<sub>es</sub> th<sub>roug</sub>h <sub>a</sub> l<sub>a</sub>t<sub>en</sub>t <sub>m</sub>i<sub>x</sub>t<sub>ure</sub> <sub>w</sub>hil<sub>e</sub> <sub>up</sub>d<sub>a</sub>ti<sub>ng</sub> <sub>an</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>pos</sub>t<sub>e</sub>ri<sub>o</sub>r<sub>.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t LAMA <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> M<sub>a</sub>rk<sub>ov</sub> DSIC <sub>a</sub>nd IR<sub>, a</sub>nd <sub>ac</sub>hi<sub>eves</sub> <sub>near-op</sub>ti<sub>ma</sub>l KL<sub>-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d <sub>we</sub>lf<sub>are.</sub> W<sub>e</sub> f<sub>ur</sub>th<sub>er</sub> d<sub>eve</sub>l<sub>op</sub> <sub>a</sub> l<sub>earn</sub>i<sub>ng-</sub>b<sub>ase</sub>d i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>recons</sub>t<sub>ruc</sub>t<sub>s</sub> th<sub>e requ</sub>i<sub>re</sub>d <sub>repor</sub>t<sub>s on</sub>li<sub>ne</sub> f<sub>rom</sub> l<sub>earne</sub>d l<sub>oca</sub>l <sub>a</sub>d<sub>van</sub>t<sub>ages an</sub>d <sub>roo</sub>t <sub>va</sub>l<sub>ues.</sub> P<sub>roo</sub>f<sub>-</sub> <sub>o</sub>f<sub>-concep</sub>t <sub>exper</sub>i<sub>men</sub>t<sub>s</sub> <sub>on</sub> <sub>rea</sub>l<sub>-wor</sub>ld <sub>commerc</sub>i<sub>a</sub>l<sub>-searc</sub>h <sub>query</sub> <sub>sp</sub>lit<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t LAMA im<sub>p</sub>r<sub>oves p</sub>l<sub>a</sub>tf<sub>o</sub>rm <sub>we</sub>lf<sub>a</sub>r<sub>e a</sub>nd r<sub>eve</sub>n<sub>ue</sub> while maintainin<sub>g</sub> user-facin<sub>g</sub> res<sub>p</sub>onse <sub>q</sub>ualit<sub>y,</sub> <sub>p</sub>rovidin<sub>g</sub> initial evidence for the feasibilit<sub>y</sub> of <sub>g</sub>eneration-native advertisin<sub>g</sub>.

## CCS Concepts

• Theory of computation → Algorithmic mechanism design;   
• Information systems → Online advertising.

## Keywords

mechanism desi<sub>g</sub>n<sub>,</sub> <sub>g</sub>ame theor<sub>y,</sub> lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models<sub>,</sub> auction<sub>,</sub> com<sub>p</sub>utational advertisin<sub>g</sub>

## 1 Introduction

“The medium is the message.”

— Marshall McLuhan, Understanding Media (1964)

The medium throu<sub>g</sub>h which <sub>p</sub>eo<sub>p</sub>le access online information is <sub>un</sub>d<sub>ergo</sub>i<sub>ng</sub> <sub>a</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> F<sub>or</sub> d<sub>eca</sub>d<sub>es,</sub> i<sub>n</sub>t<sub>erne</sub>t i<sub>n-</sub> terfaces have lar<sub>g</sub>el<sub>y</sub> or<sub>g</sub>anized information b<sub>y</sub> retrievin<sub>g,</sub> rankin<sub>g,</sub> and <sub>p</sub>resentin<sub>g</sub> content that alread<sub>y</sub> exists. Search en<sub>g</sub>ines return <sub>ran</sub>k<sub>e</sub>d <sub>resu</sub>lt<sub>s; recommen</sub>d<sub>a</sub>ti<sub>on sys</sub>t<sub>ems se</sub>l<sub>ec</sub>t it<sub>ems</sub> f<sub>rom a co</sub>l<sub>-</sub> l<sub>ec</sub>ti<sub>on.</sub> G<sub>enera</sub>ti<sub>ve</sub> AI i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> <sub>a</sub> dif<sub>eren</sub>t <sub>mo</sub>d<sub>e</sub> <sub>o</sub>f i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> access. Rather than sim<sub>p</sub>l<sub>y</sub> choosin<sub>g</sub> what to <sub>p</sub>resent<sub>,</sub> AI-native interfaces s<sub>y</sub>nthesize res<sub>p</sub>onses d<sub>y</sub>namicall<sub>y</sub> around user intent<sub>,</sub> <sub>w</sub>ith th<sub>e</sub> <sub>con</sub>t<sub>en</sub>t it<sub>se</sub>lf t<sub>a</sub>ki<sub>ng</sub> <sub>s</sub>h<sub>ape</sub> <sub>as</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> <sub>un</sub>f<sub>o</sub>ld<sub>s.</sub> Th<sub>e</sub> medium is no lon<sub>g</sub>er merel<sub>y</sub> or<sub>g</sub>anizin<sub>g</sub> information<sub>;</sub> it is <sub>g</sub>eneratin<sub>g</sub> it.

![](images/398836801f9be0e8f57d91e872a8450ddcc9f276702a9dda70a1b3fe1f884d22.jpg)  
Figure 1: In generation-native advertising, ad opportunities are no longer predefined slots: when they emerge, which advertisers are shown, and how they are presented jointly shape user experience and marketing efectiveness.

Ad<sub>ver</sub>ti<sub>s</sub>i<sub>ng</sub> h<sub>as</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>ll<sub>y</sub> b<sub>een</sub> b<sub>u</sub>ilt <sub>aroun</sub>d th<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> <sub>o</sub>f the medi<sub>u</sub>m in <sub>w</sub>hich it a<sub>pp</sub>ears. In traditional internet interfaces<sub>,</sub> thi<sub>s</sub> h<sub>as</sub> l<sub>e</sub>d t<sub>o a s</sub>i<sub>mp</sub>l<sub>e</sub> b<sub>u</sub>t <sub>power</sub>f<sub>u</sub>l <sub>a</sub>b<sub>s</sub>t<sub>rac</sub>ti<sub>on:</sub> th<sub>e s</sub>l<sub>o</sub>t<sub>.</sub> Wh<sub>e</sub>th<sub>er</sub> it is a s<sub>p</sub>onsored-search <sub>p</sub>osition<sub>,</sub> a reserved dis<sub>p</sub>la<sub>y</sub> <sub>p</sub>lacement<sub>,</sub> or a <sub>p</sub>romoted <sub>p</sub>osition in a recommendation feed<sub>,</sub> the commercial <sub>oppor</sub>t<sub>un</sub>it<sub>y can</sub> t<sub>yp</sub>i<sub>ca</sub>ll<sub>y</sub> b<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>n a</sub>d<sub>vance an</sub>d th<sub>en ass</sub>i<sub>gne</sub>d t<sub>o a</sub>d<sub>ver</sub>ti<sub>sers.</sub> B<sub>u</sub>t thi<sub>s a</sub>b<sub>s</sub>t<sub>rac</sub>ti<sub>on</sub> b<sub>ecomes</sub> l<sub>ess na</sub>t<sub>ura</sub>l <sub>w</sub>h<sub>en</sub> th<sub>e</sub> surroundin<sub>g</sub> content is itself <sub>g</sub>enerated. As shown in Fi<sub>g</sub>ure 1<sub>,</sub> a <sub>pro</sub>d<sub>uc</sub>t <sub>may</sub> b<sub>ecome</sub> <sub>re</sub>l<sub>evan</sub>t <sub>on</sub>l<sub>y</sub> <sub>as</sub> <sub>an</sub> <sub>answer</sub> d<sub>eve</sub>l<sub>ops</sub> i<sub>n</sub> <sub>a</sub> <sub>p</sub>articular direction<sub>,</sub> and an earl<sub>y g</sub>eneration choice can make a later <sup>b</sup>ran<sup>d</sup> mention eit<sup>h</sup>er natura<sup>l</sup> or jarring. In suc<sup>h</sup> settings, generation d<sub>oes more</sub> th<sub>an</sub> fill <sub>an a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng oppor</sub>t<sub>un</sub>it<sub>y—</sub>it <sub>can</sub> h<sub>e</sub>l<sub>p crea</sub>t<sub>e</sub> one. The advertisin<sub>g</sub> o<sub>pp</sub>ortunit<sub>y</sub> is no lon<sub>g</sub>er merel<sub>y</sub> a slot inside <sub>genera</sub>t<sub>e</sub>d <sub>con</sub>t<sub>en</sub>t<sub>;</sub> it <sub>can</sub> b<sub>e</sub> <sub>s</sub>h<sub>ape</sub>d b<sub>y</sub> th<sub>e</sub> <sub>genera</sub>ti<sub>on</sub> <sub>process</sub> it<sub>se</sub>lf<sub>.</sub>

This shift calls for a corres<sub>p</sub>ondin<sub>g</sub> chan<sub>g</sub>e in mechanism desi<sub>g</sub>n. Traditional advertisin<sub>g</sub> mechanisms lar<sub>g</sub>el<sub>y</sub> se<sub>p</sub>arate two <sub>q</sub>uestions: <sub>w</sub>h<sub>a</sub>t <sub>oppo</sub>rt<sub>u</sub>niti<sub>es e</sub>xi<sub>s</sub>t<sub>, a</sub>nd <sub>w</sub>h<sub>o s</sub>h<sub>ou</sub>ld r<sub>ece</sub>i<sub>ve</sub> th<sub>e</sub>m? A<sub>uc</sub>ti<sub>o</sub>n th<sub>eory</sub> h<sub>as</sub> d<sub>eve</sub>l<sub>ope</sub>d <sub>power</sub>f<sub>u</sub>l t<sub>oo</sub>l<sub>s</sub> f<sub>or</sub> th<sub>e</sub> l<sub>a</sub>tt<sub>er,</sub> t<sub>yp</sub>i<sub>ca</sub>ll<sub>y</sub> t<sub>a</sub>ki<sub>ng</sub> the former as <sub>g</sub>iven. This se<sub>p</sub>aration is natural when the advertisin<sub>g</sub> <sub>s</sub>l<sub>o</sub>t <sub>ex</sub>i<sub>s</sub>t<sub>s</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> <sub>auc</sub>ti<sub>on</sub> b<sub>eg</sub>i<sub>ns,</sub> b<sub>u</sub>t b<sub>eg</sub>i<sub>ns</sub> t<sub>o</sub> b<sub>rea</sub>k d<sub>own</sub> <sub>w</sub>h<sub>en</sub> t<sup>h</sup>e generation trajectory itse<sup>lf</sup> <sup>d</sup>etermines w<sup>h</sup>et<sup>h</sup>er an<sup>d</sup> <sup>h</sup>ow an <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> <sub>can</sub> b<sub>e</sub> <sub>mean</sub>i<sub>ng</sub>f<sub>u</sub>ll<sub>y</sub> i<sub>ncorpora</sub>t<sub>e</sub>d<sub>.</sub> Th<sub>us,</sub> <sub>a</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> th<sub>a</sub>t <sub>on</sub>l<sub>y</sub> <sub>a</sub>ll<sub>oca</sub>t<sub>es</sub> <sub>pre</sub>d<sub>e</sub>fi<sub>ne</sub>d <sub>oppor</sub>t<sub>un</sub>iti<sub>es</sub> l<sub>eaves</sub> <sub>an</sub> i<sub>mpor</sub>t<sub>an</sub>t <sub>par</sub>t <sub>o</sub>f th<sub>e</sub> <sub>genera</sub>ti<sub>ve</sub> <sub>me</sub>di<sub>um</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>process.</sub>

R<sub>ecen</sub>t <sub>wor</sub>k h<sub>as</sub> b<sub>egun</sub> t<sub>o</sub> <sub>re</sub>thi<sub>n</sub>k <sub>w</sub>h<sub>a</sub>t <sub>cons</sub>tit<sub>u</sub>t<sub>es</sub> <sub>an</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>s-</sub> in<sub>g</sub> o<sub>pp</sub>ortunit<sub>y</sub> in <sub>g</sub>enerative interfaces. Rather than rel<sub>y</sub>in<sub>g</sub> onl<sub>y</sub> <sub>on</sub> fi<sub>xe</sub>d <sub>s</sub>l<sub>o</sub>t<sub>s,</sub> <sub>some</sub> <sub>approac</sub>h<sub>es</sub> d<sub>e</sub>fi<sub>ne</sub> <sub>oppor</sub>t<sub>un</sub>iti<sub>es</sub> <sub>over</sub> <sub>segmen</sub>t<sub>s</sub> or context-de<sub>p</sub>endent <sub>p</sub>ositions within <sub>g</sub>enerated text<sub>,</sub> while others fi<sub>rs</sub>t <sub>cons</sub>t<sub>ruc</sub>t <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>responses</sub> <sub>an</sub>d th<sub>en</sub> <sub>se</sub>l<sub>ec</sub>t<sub>,</sub> <sub>ag-</sub> <sub>g</sub>re<sub>g</sub>ate<sub>,</sub> or insert s<sub>p</sub>onsored content at the res<sub>p</sub>onse level. These abstractions make advertisin<sub>g</sub> substantiall<sub>y</sub> more ada<sub>p</sub>tive to <sub>g</sub>ener-<sub>a</sub>t<sub>e</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>:</sub> <sub>w</sub>h<sub>ere</sub> <sub>an</sub> <sub>a</sub>d <sub>can</sub> <sub>appear,</sub> <sub>or</sub> <sub>w</sub>hi<sub>c</sub>h <sub>sponsore</sub>d <sub>response</sub> is a<sub>pp</sub>ro<sub>p</sub>riate<sub>,</sub> ma<sub>y</sub> now de<sub>p</sub>end on the content <sub>p</sub>roduced b<sub>y</sub> the model. Yet the mechanism t<sub>yp</sub>icall<sub>y</sub> o<sub>p</sub>erates on an o<sub>pp</sub>ortunit<sub>y</sub> that h<sub>as</sub> <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> b<sub>een</sub> id<sub>en</sub>tifi<sub>e</sub>d<sub>—a</sub> <sub>segmen</sub>t<sub>,</sub> <sub>a</sub> <sub>pos</sub>iti<sub>on,</sub> <sub>or</sub> <sub>a</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub>d res<sub>p</sub>onse. Generation sha<sub>p</sub>es the o<sub>pp</sub>ortunit<sub>y,</sub> but is not itself the <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>process.</sub>

In this <sub>p</sub>a<sub>p</sub>er<sub>,</sub> we take the next ste<sub>p</sub> and formulate a <sub>g</sub>enerationnative <sub>p</sub>aradi<sub>g</sub>m for com<sub>p</sub>utational advertisin<sub>g,</sub> in which advertisin<sub>g</sub> is directl<sub>y</sub> embedded into the <sub>g</sub>eneration <sub>p</sub>rocess. Advertisers are treated as strate<sub>g</sub>ic <sub>p</sub>artici<sub>p</sub>ants whose messa<sub>g</sub>es can influence <sup>h</sup>ow t<sup>h</sup>e response evo<sup>l</sup>ves, w<sup>h</sup>i<sup>l</sup>e t<sup>h</sup>e p<sup>l</sup>at<sup>f</sup>orm joint<sup>l</sup>y <sup>d</sup>etermines <sub>g</sub>enerated content<sub>,</sub> advertisin<sub>g</sub> allocation<sub>,</sub> and <sub>p</sub>a<sub>y</sub>ments. Within this broader <sub>p</sub>aradi<sub>g</sub>m<sub>,</sub> we stud<sub>y</sub> a <sub>p</sub>articularl<sub>y</sub> native instantiation <sub>a</sub>t th<sub>e</sub> t<sub>o</sub>k<sub>en</sub> l<sub>eve</sub>l<sub>:</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>sers</sub> <sub>may</sub> i<sub>n</sub>fl<sub>uence</sub> th<sub>e</sub> <sub>response</sub> <sub>a</sub>t <sub>every</sub> <sub>g</sub>eneration ste<sub>p</sub>. A user <sub>q</sub>uer<sub>y</sub> initializes a se<sub>q</sub>uential <sub>p</sub>rocess<sub>,</sub> and at <sub>eac</sub>h <sub>genera</sub>t<sub>e</sub>d <sub>pre</sub>fi<sub>x,</sub> th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>com</sub>bi<sub>nes</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> <sub>messages</sub> <sub>w</sub>ith th<sub>e re</sub>f<sub>erence</sub> l<sub>anguage mo</sub>d<sub>e</sub>l <sub>an</sub>d th<sub>e curren</sub>t <sub>mec</sub>h<sub>an</sub>i<sub>sm s</sub>t<sub>a</sub>t<sub>e</sub> t<sub>o</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne w</sub>h<sub>a</sub>t t<sub>o</sub>k<sub>en</sub> t<sub>o genera</sub>t<sub>e nex</sub>t <sub>an</sub>d h<sub>ow</sub> th<sub>e a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on s</sub>h<sub>ou</sub>ld <sub>evo</sub>l<sub>ve.</sub> Th<sub>e goa</sub>l i<sub>s</sub> t<sub>o</sub> i<sub>ncorpora</sub>t<sub>e a</sub>d<sub>ver</sub>ti<sub>ser</sub> i<sub>ncen-</sub> tives and <sub>p</sub>latform monetization into <sub>g</sub>eneration while <sub>p</sub>reservin<sub>g</sub> th<sub>e na</sub>t<sub>ura</sub>l<sub>ness an</sub>d <sub>use</sub>f<sub>u</sub>l<sub>ness o</sub>f th<sub>e genera</sub>t<sub>e</sub>d <sub>con</sub>t<sub>en</sub>t<sub>.</sub>

Makin<sub>g</sub> advertisin<sub>g</sub> a native <sub>p</sub>art of<sub>g</sub>eneration introduces several technical challen<sub>g</sub>es. Most im<sub>p</sub>ortantl<sub>y,</sub> incentive desi<sub>g</sub>n becomes i<sub>n</sub>h<sub>eren</sub>tl<sub>y sequen</sub>ti<sub>a</sub>l<sub>: a repor</sub>t <sub>su</sub>b<sub>m</sub>itt<sub>e</sub>d <sub>a</sub>t <sub>one pre</sub>fi<sub>x can c</sub>h<sub>ange</sub> t<sup>h</sup>e <sup>f</sup>uture generation trajectory, w<sup>h</sup>ic<sup>h</sup> in turn c<sup>h</sup>anges t<sup>h</sup>e a<sup>d</sup>- <sub>v</sub>ertiser’s s<sub>u</sub>bse<sub>qu</sub>ent o<sub>pp</sub>ort<sub>u</sub>nities and e<sub>v</sub>ent<sub>u</sub>al <sub>p</sub>a<sub>y</sub>of. Tr<sub>u</sub>thf<sub>u</sub>l b<sub>e</sub>h<sub>av</sub>i<sub>or mus</sub>t th<sub>ere</sub>f<sub>ore rema</sub>i<sub>n op</sub>ti<sub>ma</sub>l <sub>no</sub>t <sub>on</sub>l<sub>y a</sub>t th<sub>e</sub> b<sub>eg</sub>i<sub>nn</sub>i<sub>ng o</sub>f <sub>genera</sub>ti<sub>on,</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> <sub>a</sub>ft<sub>er</sub> <sub>any</sub> <sub>reac</sub>h<sub>a</sub>bl<sub>e</sub> f<sub>u</sub>t<sub>ure</sub> <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> Ad<sub>ver</sub>ti<sub>ser</sub> i<sub>n-</sub> fl<sub>uence</sub> <sub>mus</sub>t <sub>a</sub>l<sub>so</sub> b<sub>e</sub> b<sub>a</sub>l<sub>ance</sub>d <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> <sub>re</sub>f<sub>erence</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l <sub>so</sub> th<sub>a</sub>t <sub>mone</sub>ti<sub>za</sub>ti<sub>on</sub> d<sub>oes</sub> <sub>no</sub>t <sub>come</sub> <sub>a</sub>t th<sub>e</sub> <sub>expense</sub> <sub>o</sub>f <sub>response</sub> <sub>q</sub>ualit<sub>y</sub>. At the same time<sub>,</sub> lan<sub>g</sub>ua<sub>g</sub>e model inference alread<sub>y</sub> incurs substantial servin<sub>g</sub> latenc<sub>y,</sub> re<sub>q</sub>uirin<sub>g</sub> the mechanism to introduce <sub>on</sub>l<sub>y m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>a</sub>dditi<sub>ona</sub>l <sub>over</sub>h<sub>ea</sub>d<sub>.</sub> Fi<sub>na</sub>ll<sub>y, a</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e mec</sub>h<sub>an</sub>i<sub>sm</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>s</sub>till <sub>a</sub>d<sub>m</sub>it <sub>a</sub> <sub>mean</sub>i<sub>ng</sub>f<sub>u</sub>l <sub>e</sub>fi<sub>c</sub>i<sub>ency</sub> <sub>guaran</sub>t<sub>ee</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> <sub>an</sub> id<sub>ea</sub>l <sub>we</sub>lf<sub>are-max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>.</sub>

T<sub>o a</sub>dd<sub>ress</sub> th<sub>ese c</sub>h<sub>a</sub>ll<sub>enges, we propose</sub> th<sub>e</sub> L<sub>a</sub>t<sub>en</sub>t Ad<sub>ver</sub>ti<sub>ser</sub> Mixture Auction (LAMA) for the fundamental sin<sub>g</sub>le-winner settin<sub>g</sub>. At <sub>eac</sub>h <sub>pre</sub>fi<sub>x, a</sub>d<sub>ver</sub>ti<sub>sers</sub>’ l<sub>oca</sub>l <sub>repor</sub>t<sub>s</sub> i<sub>n</sub>d<sub>uce a</sub>d<sub>ver</sub>ti<sub>ser-spec</sub>ifi<sub>c</sub> n<sub>e</sub>xt-t<sub>o</sub>k<sub>e</sub>n <sub>po</sub>li<sub>c</sub>i<sub>es, a</sub>nd LAMA <sub>ge</sub>n<sub>e</sub>r<sub>a</sub>t<sub>es</sub> n<sub>e</sub>xt t<sub>o</sub>k<sub>e</sub>n fr<sub>o</sub>m <sub>a</sub> l<sub>a</sub>t<sub>e</sub>nt mixture of these <sub>p</sub>olicies. The mixture wei<sub>g</sub>hts evolve as a Ba<sub>y</sub>esian <sub>a</sub>ll<sub>oca</sub>ti<sub>on pos</sub>t<sub>er</sub>i<sub>or as</sub> t<sub>o</sub>k<sub>ens are o</sub>b<sub>serve</sub>d<sub>,</sub> li<sub>n</sub>ki<sub>ng</sub> th<sub>e genera</sub>ti<sub>on</sub> trajectory to t<sup>h</sup>e eventua<sup>l</sup> a<sup>d</sup>vertising a<sup>ll</sup>ocation. Once generation t<sub>erm</sub>i<sub>na</sub>t<sub>es,</sub> thi<sub>s pos</sub>t<sub>er</sub>i<sub>or</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> t<sub>oge</sub>th<sub>er</sub> with <sub>p</sub>a<sub>y</sub>ments desi<sub>g</sub>ned for the se<sub>q</sub>uential settin<sub>g</sub>. We show that LAMA <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> M<sub>a</sub>rk<sub>ov</sub> d<sub>o</sub>min<sub>a</sub>nt-<sub>s</sub>tr<sub>a</sub>t<sub>egy</sub> in<sub>ce</sub>nti<sub>ve</sub> <sub>co</sub>m<sub>pa</sub>tibil it<sub>y</sub> and indi<sub>v</sub>id<sub>u</sub>al rationalit<sub>y</sub>. Moreo<sub>v</sub>er<sub>,</sub> it achie<sub>v</sub>es near-o<sub>p</sub>timal KL-re<sub>g</sub>ularized welfare<sub>,</sub> with the o<sub>p</sub>timalit<sub>y</sub> <sub>g</sub>a<sub>p</sub> vanishin<sub>g</sub> as the r<sub>egu</sub>l<sub>a</sub>riz<sub>a</sub>ti<sub>o</sub>n dimini<sub>s</sub>h<sub>es.</sub>

W<sub>e</sub> f<sub>ur</sub>th<sub>er</sub> d<sub>eve</sub>l<sub>op a prac</sub>ti<sub>ca</sub>l i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>avo</sub>id<sub>s ex-</sub> <sub>p</sub>licitl<sub>y</sub> reasonin<sub>g</sub> over the full token tree or re<sub>q</sub>uirin<sub>g</sub> advertisers t<sub>o</sub> <sub>compu</sub>t<sub>e</sub> <sub>comp</sub>l<sub>ex</sub> <sub>repor</sub>t<sub>s</sub> th<sub>emse</sub>l<sub>ves.</sub> Th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>prov</sub>id<sub>es</sub> <sub>re-</sub> <sub>por</sub>ti<sub>ng</sub> <sub>as</sub> <sub>a</sub> l<sub>earne</sub>d <sub>serv</sub>i<sub>ce:</sub> <sub>a</sub> <sub>s</sub>h<sub>are</sub>d <sub>a</sub>d<sub>ver</sub>ti<sub>ser-con</sub>diti<sub>one</sub>d <sub>mo</sub>d<sub>e</sub>l <sub>recovers</sub> l<sub>oca</sub>l <sub>s</sub>i<sub>gna</sub>l<sub>s a</sub>l<sub>ong</sub> th<sub>e rea</sub>li<sub>ze</sub>d <sub>genera</sub>ti<sub>on pa</sub>th <sub>an</sub>d <sub>an-</sub> <sub>c</sub>h<sub>ors</sub> th<sub>em w</sub>ith <sub>an a</sub>d<sub>ver</sub>ti<sub>ser-spec</sub>ifi<sub>c roo</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>e, a</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> r<sub>epo</sub>rt<sub>s</sub> r<sub>equ</sub>ir<sub>e</sub>d b<sub>y</sub> LAMA t<sub>o</sub> b<sub>e</sub> r<sub>eco</sub>n<sub>s</sub>tr<sub>uc</sub>t<sub>e</sub>d <sub>o</sub>nlin<sub>e.</sub> Thi<sub>s</sub> d<sub>e</sub>- si<sub>g</sub>n turns the theoretical se<sub>q</sub>uential mechanism into a li<sub>g</sub>htwei<sub>g</sub>ht <sub>serv</sub>i<sub>ng-</sub>ti<sub>me proce</sub>d<sub>ure</sub> th<sub>a</sub>t <sub>can opera</sub>t<sub>e over a sma</sub>ll <sub>can</sub>did<sub>a</sub>t<sub>e se</sub>t <sub>o</sub>f <sub>a</sub>d<sub>ver</sub>ti<sub>sers</sub> <sub>w</sub>ith <sub>mo</sub>d<sub>es</sub>t <sub>a</sub>dditi<sub>ona</sub>l l<sub>anguage-mo</sub>d<sub>e</sub>l <sub>compu</sub>t<sub>a</sub>ti<sub>on.</sub>

We e<sub>v</sub>al<sub>u</sub>ate LAMA in sim<sub>u</sub>lation on real-<sub>w</sub>orld <sub>qu</sub>er<sub>y</sub> s<sub>p</sub>lits <sub>w</sub>ith model-<sub>p</sub>redicted im<sub>p</sub>ression values. We com<sub>p</sub>are a<sub>g</sub>ainst heuristic <sub>mec</sub>h<sub>an</sub>i<sub>sms</sub> th<sub>a</sub>t <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng</sub> <sub>e</sub>ith<sub>er</sub> b<sub>e</sub>f<sub>ore</sub> <sub>or</sub> <sub>a</sub>ft<sub>er</sub> <sub>genera-</sub> tion<sub>,</sub> as well as a re<sub>p</sub>resentative res<sub>p</sub>onse-level a<sub>gg</sub>re<sub>g</sub>ation mecha-<sub>n</sub>i<sub>sm.</sub> A<sub>cross</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>se</sub>tti<sub>ngs,</sub> LAMA <sub>ac</sub>hi<sub>eves</sub> th<sub>e</sub> <sub>s</sub>t<sub>ronges</sub>t <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>we</sub>lf<sub>are</sub> <sub>an</sub>d <sub>revenue</sub> <sub>w</sub>hil<sub>e</sub> <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> hi<sub>g</sub>h <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> value and user-side res<sub>p</sub>onse <sub>q</sub>ualit<sub>y</sub>. These results <sub>p</sub>rovide an initial <sub>p</sub>roof of conce<sub>p</sub>t for the <sub>g</sub>eneration-native advertisin<sub>g</sub> <sub>p</sub>aradi<sub>g</sub>m: allowin<sub>g</sub> advertisers to <sub>p</sub>artici<sub>p</sub>ate directl<sub>y</sub> in the token-level <sub>g</sub>eneration <sub>p</sub>rocess can im<sub>p</sub>rove monetization while <sub>p</sub>reservin<sub>g</sub> hi<sub>g</sub>h-<sub>q</sub>ualit<sub>y g</sub>enerated res<sub>p</sub>onses.

O<sub>ur con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons are summar</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows.</sub>

(1) Generation-native advertising. We develop a token-level formulation of <sub>g</sub>eneration-native advertisin<sub>g,</sub> where advertiser <sub>p</sub>artici<sub>p</sub>ation is embedded directl<sub>y</sub> into the se<sub>q</sub>uential generation process an<sup>d</sup> joint<sup>l</sup>y s<sup>h</sup>apes t<sup>h</sup>e response an<sup>d</sup> <sub>even</sub>t<sub>ua</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on.</sub>

(2) Token-level mechanism design. We propose LAMA for th<sub>e</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>l <sub>s</sub>i<sub>ng</sub>l<sub>e-w</sub>i<sub>nner</sub> <sub>se</sub>tti<sub>ng</sub> <sub>an</sub>d <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h M<sub>ar</sub>k<sub>ov</sub> DSIC and IR<sub>,</sub> to<sub>g</sub>ether <sub>w</sub>ith near-o<sub>p</sub>timal KL-re<sub>gu</sub>larized welfare, with an additive <sub>g</sub>a<sub>p</sub> of at most � lo<sub>g</sub> |N |.

(3) Practical implementation. We develop a learning-based im<sub>p</sub>lementation that reconstructs the re<sub>p</sub>orts re<sub>q</sub>uired b<sub>y</sub> LAMA online and s<sub>upp</sub>orts eficient servin<sub>g</sub>.

(4) Proof-of-concept evaluation. Experiments on real-world search <sub>q</sub>uer<sub>y</sub> s<sub>p</sub>lits show that LAMA im<sub>p</sub>roves <sub>p</sub>latform <sub>we</sub>lf<sub>are an</sub>d <sub>revenue w</sub>hil<sub>e ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng a</sub>d<sub>ver</sub>ti<sub>ser va</sub>l<sub>ue</sub> and user-facin<sub>g</sub> res<sub>p</sub>onse <sub>q</sub>ualit<sub>y</sub>.

## 2 Model

W<sub>e mo</sub>d<sub>e</sub>l t<sub>o</sub>k<sub>en-</sub>l<sub>eve</sub>l <sub>a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng as a sequen</sub>ti<sub>a</sub>l <sub>mec</sub>h<sub>an</sub>i<sub>sm em-</sub> b<sub>e</sub>dd<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>genera</sub>ti<sub>on</sub> <sub>process</sub> <sub>o</sub>f <sub>a</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> A <sub>user</sub> <sub>query</sub> initi<sub>a</sub>liz<sub>es a</sub> t<sub>o</sub>k<sub>e</sub>n-l<sub>eve</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n <sub>p</sub>r<sub>ocess.</sub> At <sub>eac</sub>h n<sub>o</sub>n-t<sub>e</sub>rmin<sub>a</sub>l <sub>pre</sub>fi<sub>x,</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>sers</sub> <sub>may</sub> <sub>su</sub>b<sub>m</sub>it l<sub>oca</sub>l <sub>repor</sub>t<sub>s</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> <sub>pu</sub>bli<sub>c</sub>l<sub>y</sub> <sub>o</sub>b<sub>serve</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> <sub>an</sub>d th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>uses</sub> th<sub>ese</sub> <sub>repor</sub>t<sub>s</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith it<sub>s</sub> <sub>pr</sub>i<sub>va</sub>t<sub>e</sub>l<sub>y</sub> <sub>o</sub>b<sub>serve</sub>d hi<sub>s</sub>t<sub>ory</sub> t<sub>o</sub> <sub>c</sub>h<sub>oose</sub> th<sub>e</sub> <sub>nex</sub>t t<sub>o</sub>k<sub>en</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on,</sub> th<sub>e</sub> advertisin<sub>g</sub> allocation<sub>,</sub> and <sub>p</sub>a<sub>y</sub>ments. The central distinction from <sub>c</sub>l<sub>ass</sub>i<sub>ca</sub>l <sub>sponsore</sub>d<sub>-searc</sub>h <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> i<sub>s</sub> th<sub>a</sub>t th<sub>e a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng oppor</sub>t<sub>u</sub> nit<sub>y</sub> is not an exo<sub>g</sub>enousl<sub>y g</sub>iven slot. Instead<sub>,</sub> it is endo<sub>g</sub>enousl<sub>y</sub> s<sup>h</sup>ape<sup>d b</sup>y t<sup>h</sup>e generate<sup>d</sup> trajectory itse<sup>lf</sup>.

Context space and reference generation. Let V be the token vo-<sub>ca</sub>b<sub>u</sub>l<sub>ary</sub> <sub>an</sub>d l<sub>e</sub>t � b<sub>e</sub> th<sub>e</sub> <sub>max</sub>i<sub>mum</sub> <sub>genera</sub>ti<sub>on</sub> l<sub>eng</sub>th<sup>1</sup><sub>.</sub> Th<sub>e</sub> <sub>con</sub>t<sub>ex</sub>t s<sub>p</sub>ace <sup>i</sup>s $S : = \mathcal { V } ^ { \leq L } = \bigcup _ { t = 0 } ^ { L } \mathcal { V } ^ { t }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\mathcal { V } ^ { 0 } = \{ \boldsymbol { \emptyset } \}$ . For two contexts $x _ { 1 } , x _ { 2 } \in S _ { }$ <sub>,</sub> we write $[ x _ { 1 } , x _ { 2 } ]$ f<sub>or</sub> th<sub>e</sub>i<sub>r</sub> <sub>conca</sub>t<sub>ena</sub>ti<sub>on</sub> <sub>w</sub>h<sub>enever</sub> th<sub>e</sub> resultin<sub>g</sub> se<sub>q</sub>uence has len<sub>g</sub>th at most �. A user <sub>q</sub>uer<sub>y</sub> $q \in S$ i<sub>s</sub> drawn from a <sub>q</sub>uer<sub>y</sub> distribution D and serves as the initial state.

The <sub>p</sub>latform has an or<sub>g</sub>anic lan<sub>g</sub>ua<sub>g</sub>e-model <sub>p</sub>olic<sub>y</sub> $\pi _ { \mathrm { r e f } } ( \cdot \mid s )$ ∈ $\Delta _ { + + } ( \mathcal { V } ) , s \in S ,$ <sub>, w</sub>hi<sub>c</sub>h d<sub>escr</sub>ib<sub>es</sub> th<sub>e nex</sub>t<sub>-</sub>t<sub>o</sub>k<sub>en</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> <sub>a</sub>b<sub>se</sub>n<sub>ce o</sub>f <sub>a</sub>d<sub>ve</sub>rti<sub>s</sub>in<sub>g</sub> int<sub>e</sub>r<sub>ve</sub>nti<sub>o</sub>n<sup>2</sup><sub>.</sub> W<sub>e a</sub>l<sub>so w</sub>rit<sub>e</sub> $\pi _ { 0 }$ f<sub>or</sub> $\pi _ { \mathrm { r e f } } . \mathrm { A }$ trajectory <sup>f</sup>rom query � is $\tau = ( s _ { 0 } = q , a _ { 0 } , s _ { 1 } , a _ { 1 } , . . . , s _ { T _ { \tau } } ) ,$ �<sub>�+1</sub> = $\left[ s _ { t } , a _ { t } \right]$ . Let $\mathsf { E O S } \in \mathcal { N }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e en</sub>d<sub>-o</sub>f<sub>-sequence</sub> t<sub>o</sub>k<sub>en.</sub> A <sub>s</sub>t<sub>a</sub>t<sub>e</sub> is terminal if it contains EOS or reaches the maximum len<sub>g</sub>th. We d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>se</sub>t b<sub>y</sub> $s _ { T }$ <sub>, an</sub>d <sub>wr</sub>it<sub>e</sub> $L ( s ) \subseteq S _ { T }$ f<sub>or</sub> th<sub>e se</sub>t <sub>o</sub>f termina<sup>l</sup> <sup>d</sup>escen<sup>d</sup>ants o<sup>f</sup> a non-termina<sup>l</sup> state �. For a trajectory �, it<sub>s rea</sub>li<sub>ze</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s</sub> d<sub>eno</sub>t<sub>e</sub>d b<sub>y</sub> $\ell ( \tau ) = s _ { T _ { \tau } }$

Advertisers and private types. There are $n \geq 2$ <sub>a</sub>d<sub>ve</sub>rti<sub>se</sub>r<sub>s,</sub> ind<sub>exe</sub>d b<sub>y</sub> $N = [ n ] = \{ 1 , . . . , n \}$ <sub>.</sub> E<sub>ac</sub>h <sub>a</sub>d<sub>ve</sub>rti<sub>se</sub>r � h<sub>as</sub> <sub>a</sub> <sub>p</sub>ri<sub>va</sub>t<sub>e</sub> t<sub>e</sub>rmi-<sub>na</sub>l <sub>rewar</sub>d f<sub>unc</sub>ti<sub>on</sub> $r _ { i } : S _ { T }  \mathbb { R } _ { \geq 0 }$ <sub>.</sub> Th<sub>e va</sub>l<sub>ue</sub> $r _ { i } ( \ell )$ re<sub>p</sub>resents a<sup>d</sup>- vertiser $i ^ { \prime } s$ value from receivin<sub>g</sub> the advertisin<sub>g</sub> o<sub>pp</sub>ortunit<sub>y</sub> when the <sub>g</sub>enerated res<sub>p</sub>onse terminates at ℓ. We write $r = ( r _ { 1 } , \ldots . r _ { n } )$ f<sub>or</sub> th<sub>e</sub> t<sub>ype pro</sub>fil<sub>e an</sub>d $r _ { - i }$ f<sub>or</sub> th<sub>e pro</sub>fil<sub>e exc</sub>l<sub>u</sub>di<sub>ng a</sub>d<sub>ver</sub>ti<sub>ser</sub> �<sub>.</sub>

The allocation of the advertisin<sub>g</sub> o<sub>pp</sub>ortunit<sub>y</sub> is re<sub>p</sub>resented b<sub>y</sub> a vector $\lambda \in \Lambda \subseteq [ 0 , 1 ] ^ { n }$ , where Λ is an exo<sub>g</sub>enousl<sub>y g</sub>iven feasible al location set. For exam<sub>p</sub>le, Λ ma<sub>y</sub> encode slot-ca<sub>p</sub>acit<sub>y</sub> constraints or ex<sub>p</sub>osure constraints. This <sub>p</sub>a<sub>p</sub>er focuses on the fundamental sin<sub>g</sub>lewinner settin<sub>g</sub>: the advertisin<sub>g</sub> o<sub>pp</sub>ortunit<sub>y</sub> is endo<sub>g</sub>enousl<sub>y</sub> sha<sub>p</sub>ed b<sub>y</sub> th<sub>e</sub> <sub>genera</sub>ti<sub>on</sub> <sub>process,</sub> b<sub>u</sub>t th<sub>e</sub> <sub>even</sub>t<sub>ua</sub>l <sub>sponsore</sub>d <sub>ou</sub>t<sub>come</sub> i<sub>s</sub> assi<sub>g</sub>ned to a sin<sub>g</sub>le advertiser. Thus<sub>,</sub> $\Lambda = \Delta ( N )$ <sub>.</sub> Th<sub>e</sub> <sub>rea</sub>li<sub>ze</sub>d <sub>gross</sub> va<sup>l</sup>ue o<sup>f</sup> a<sup>d</sup>vertiser � on trajectory � is t<sup>h</sup>ere<sup>f</sup>ore $\lambda _ { i } r _ { i } ( \ell ( \tau ) )$ ).

Platform histories. To describe a mechanism that operates during <sub>g</sub>eneration<sub>,</sub> we introduce the histor<sub>y p</sub>rivatel<sub>y</sub> observed b<sub>y</sub> the <sub>p</sub>latform. A <sub>p</sub>latform histor<sub>y</sub> at time � is

$$
h _ { t } = ( s _ { 0 } , m _ { 0 } , a _ { 0 } , \ldots , s _ { t - 1 } , m _ { t - 1 } , a _ { t - 1 } , s _ { t } ) ,
$$

<sub>w</sub>h<sub>ere</sub> $s _ { 0 } = q , a _ { t } \in \mathcal { V }$ i<sub>s</sub> th<sub>e</sub> t<sub>o</sub>k<sub>en genera</sub>t<sub>e</sub>d <sub>a</sub>t ti<sub>me</sub> �<sub>, an</sub>d $\begin{array} { r l } { m _ { t } } & { { } = } \end{array}$ $( m _ { 1 , t } , \ldots , m _ { n , t } )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e pro</sub>fil<sub>e o</sub>f <sub>messages su</sub>b<sub>m</sub>itt<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>sers</sub> <sub>a</sub>t ti<sub>me</sub> �<sub>.</sub> Ad<sub>ver</sub>ti<sub>sers</sub> <sub>o</sub>b<sub>serve</sub> <sub>on</sub>l<sub>y</sub> th<sub>e</sub> <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { t } ,$ <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>pu</sub>bli<sub>c,</sub> <sub>w</sub>h<sub>ereas</sub> th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>o</sub>b<sub>serves</sub> th<sub>e</sub> <sub>en</sub>ti<sub>re</sub> hi<sub>s</sub>t<sub>ory.</sub> The state evolves deterministicall<sub>y</sub> accordin<sub>g</sub> to $s _ { t + 1 } = \left[ s _ { t } , a _ { t } \right]$ . Let H denote the set of non-terminal <sub>p</sub>latform histories, and let $s ( h )$ b<sub>e</sub> th<sub>e curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e a</sub>t th<sub>e en</sub>d <sub>o</sub>f hi<sub>s</sub>t<sub>ory</sub> ℎ<sub>.</sub> A t<sub>erm</sub>i<sub>na</sub>l <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> hi<sub>s</sub>t<sub>ory</sub> i<sub>s</sub> d<sub>eno</sub>t<sub>e</sub>d b<sub>y</sub> $h _ { T } = ( s _ { 0 } , m _ { 0 } , a _ { 0 } , \ldots , s _ { T - 1 } , m _ { T - 1 } , a _ { T - 1 } , s _ { T } )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $s _ { T } \in$ $\boldsymbol { s _ { T } }$ . We <sub>w</sub>rite $\tau ( h _ { T } )$ <sup>f</sup>or t<sup>h</sup>e to<sup>k</sup>en trajectory in<sup>d</sup>uce<sup>d b</sup>y t<sup>h</sup>e termina<sup>l</sup> histor<sub>y</sub> $h _ { T } ,$ <sub>an</sub>d $\ell ( h _ { T } ) = s _ { T }$ f<sub>or</sub> it<sub>s</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e.</sub>

Sequential mechanisms. A sequential mechanism for token-level advertisin<sub>g</sub> is a tu<sub>p</sub>le

$$
\Gamma = \big ( \{ M _ { i } ( h ) \} _ { i \in N , h \in \mathcal { H } } , x , \lambda , p \big ) .
$$

F<sub>or eac</sub>h <sub>non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> hi<sub>s</sub>t<sub>ory</sub> $h , \ M _ { i } ( h )$ i<sub>s a</sub>d<sub>ve</sub>rti<sub>se</sub>r $i ^ { \prime } s$ feasible messa<sub>g</sub>e s<sub>p</sub>ace after that histor<sub>y</sub>. Given $h \in { \mathcal { H } }$ an<sup>d</sup> a joint <sup>messa</sup>g<sup>e</sup> $m = ( m _ { 1 } , . . . , m _ { n } ) \in \mathcal { M } ( h ) : = \prod _ { i \in \mathcal { N } } \mathcal { M } _ { i } ( h )$ <sub>,</sub> th<sub>e nex</sub>t<sub>-</sub> t<sub>o</sub>k<sub>en ru</sub>l<sub>e</sub> i<sub>s</sub> $x ( \cdot \mid h , m ) \in \Delta ( \mathcal { V } )$ <sub>.</sub> The instantaneo<sub>u</sub>s <sub>p</sub>a<sub>y</sub>ment r<sub>u</sub>le i<sub>s</sub> $p _ { i } ( h , m , a ) \ \in \ \mathbb { R } ,$ <sub>, w</sub>hi<sub>c</sub>h <sub>spec</sub>ifi<sub>es a</sub>d<sub>ver</sub>ti<sub>ser</sub> �’<sub>s paymen</sub>t <sub>w</sub>h<sub>en</sub> t<sub>o</sub>k<sub>en</sub> <sub>�</sub> i<sub>s</sub> <sub>genera</sub>t<sub>e</sub>d <sub>a</sub>ft<sub>er</sub> hi<sub>s</sub>t<sub>ory</sub> ℎ <sub>un</sub>d<sub>er</sub> <sub>repor</sub>t <sub>pro</sub>fil<sub>e</sub> <sub>�.</sub> Th<sub>e</sub> fi<sub>na</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on ru</sub>l<sub>e</sub> i<sub>s</sub> $\lambda ( h _ { T } ) \in \Lambda$ <sub>.</sub> Th<sub>us,</sub> th<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> <sub>message</sub> <sub>spaces,</sub> t<sub>o</sub>k<sub>en-genera</sub>ti<sub>on ru</sub>l<sub>e, paymen</sub>t<sub>s, an</sub>d fi<sub>na</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on may</sub> d<sub>epen</sub>d on the entire <sub>p</sub>latform histor<sub>y</sub><sup>3</sup>. In <sub>p</sub>articular<sub>,</sub> histor<sub>y</sub>-de<sub>p</sub>endent <sub>message spaces a</sub>ll<sub>ow</sub> th<sub>e p</sub>l<sub>a</sub>tf<sub>orm</sub> t<sub>o c</sub>h<sub>ec</sub>k <sub>a new repor</sub>t <sub>aga</sub>i<sub>ns</sub>t values <sub>p</sub>reviousl<sub>y</sub> recorded in its re<sub>p</sub>ort led<sub>g</sub>er.

The timin<sub>g</sub> is as follows.

(1) The user <sub>q</sub>uer<sub>y</sub> $q \sim \mathcal { D }$ initi<sub>a</sub>liz<sub>es</sub> th<sub>e p</sub>r<sub>ocess a</sub>t $s _ { 0 } = q .$

(2) At each non-terminal state $s _ { t } ,$ <sub>eac</sub>h <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> � <sub>su</sub>b<sub>m</sub>it<sub>s a</sub> <sup>messa</sup>g<sup>e</sup> $m _ { i , t } \in \mathcal { M } _ { i } ( h _ { t } )$ . Let $m _ { t } = ( m _ { 1 , t } , \ldots , m _ { n , t } )$

(3) The mechanism chooses a next-token distribution $x ( \cdot \ |$ $h _ { t } , m _ { t } )$ <sub>,</sub> d<sub>raws</sub> $a _ { t } \sim x ( { \cdot } \mid h _ { t } , m _ { t } )$ <sub>, an</sub>d <sub>c</sub>h<sub>arges</sub> $p _ { i } ( h _ { t } , m _ { t } , a _ { t } )$ f<sub>or eac</sub>h $i \in N$

(4) The state u<sub>p</sub>dates deterministicall<sub>y</sub> as $s _ { t + 1 } ~ = ~ \left[ s _ { t } , a _ { t } \right]$ <sub>.</sub> If $s _ { t + 1 } \in S _ { T }$ <sub>,</sub> th<sub>e process</sub> t<sub>erm</sub>i<sub>na</sub>t<sub>es an</sub>d th<sub>e mec</sub>h<sub>an</sub>i<sub>sm</sub> i<sub>m-</sub> <sub>p</sub>l<sub>emen</sub>t<sub>s</sub> th<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> $\lambda ( h _ { T } ) \in \Lambda$

Di<sub>rec</sub>t <sub>mec</sub>h<sub>an</sub>i<sub>sms are a spec</sub>i<sub>a</sub>l <sub>case as</sub> th<sub>e p</sub>l<sub>a</sub>tf<sub>orm can as</sub>k <sub>eac</sub>h <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> t<sub>o su</sub>b<sub>m</sub>it it<sub>s en</sub>ti<sub>re rewar</sub>d f<sub>unc</sub>ti<sub>on.</sub> H<sub>owever, suc</sub>h <sub>mec</sub>h<sub>an</sub>i<sub>sms are</sub> i<sub>mprac</sub>ti<sub>ca</sub>l f<sub>or on</sub>li<sub>ne genera</sub>ti<sub>on</sub> b<sub>ecause</sub> f<sub>u</sub>ll <sub>re-</sub> <sub>p</sub>ortin<sub>g</sub> ma<sub>y</sub> be slow and ma<sub>y</sub> reveal <sub>p</sub>rivate information about trajectories t<sup>h</sup>at are never rea<sup>l</sup>ize<sup>d</sup>. Our <sup>d</sup>e<sup>fi</sup>nition t<sup>h</sup>ere<sup>f</sup>ore em-<sub>p</sub>h<sub>as</sub>i<sub>zes</sub> li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht <sub>repor</sub>t<sub>s,</sub> <sub>an</sub>d <sub>we</sub> <sub>w</sub>ill <sub>see</sub> th<sub>a</sub>t th<sub>ey</sub> <sub>su</sub>fi<sub>ce</sub> t<sub>o</sub> <sub>ac</sub>hi<sub>eve</sub> <sub>s</sub>t<sub>rong</sub> <sub>per</sub>f<sub>ormance</sub> f<sub>or</sub> th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm.</sub>

Strategies and induced policies. Let $\Theta _ { i }$ d<sub>eno</sub>t<sub>e</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> �’<sub>s</sub> t<sub>ype</sub> s<sub>p</sub>ace. A strate<sub>gy</sub> ofadvertiser � ma<sub>p</sub>s its t<sub>yp</sub>e and each non-terminal <sub>p</sub>ublic state to a messa<sub>g</sub>e: $\sigma _ { i } : \Theta _ { i } { \times } ( S \backslash S _ { T } ) \to \bigcup _ { h \in \mathcal { H } } M _ { i } ( h )$ <sub>.</sub> A <sub>s</sub>t<sub>ra</sub>t<sub>-</sub> $\mathrm { e g y }$ i<sub>s</sub> f<sub>eas</sub>ibl<sub>e</sub> if $\sigma _ { i } ( r _ { i } , s ( h ) ) \in M _ { i } ( h )$ at ever<sub>y</sub> histor<sub>y</sub> at which it is <sub>use</sub>d<sub>.</sub> Gi<sub>ve</sub>n $\sigma = ( \sigma _ { 1 } , \ldots , \sigma _ { n } )$ an<sup>d</sup> type pro<sup>fil</sup>e �, t<sup>h</sup>e in<sup>d</sup>uce<sup>d</sup> joint re<sub>p</sub>ort at histor<sub>y</sub> ℎ is $m ^ { \sigma , r } ( h ) \ : = \ ( \sigma _ { 1 } ( r _ { 1 } , s ( h ) ) , \ldots , \sigma _ { n } ( r _ { n } , s ( h ) ) )$ To<sub>g</sub>ether with the mechanism Γ, these re<sub>p</sub>orts determine the continuation token <sub>p</sub>olic<sub>y</sub> $\pi _ { \sigma , r } ^ { \Gamma } ( a \mid h ) : = x \left( a \mid h , m ^ { \sigma , r } ( h ) \right) , a \in \mathcal { V }$ <sub>,</sub> <sub>an</sub>d h<sub>ence a</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\mathbb { P } _ { \sigma , r } ^ { \Gamma } ( \cdot \mid h )$ over terminal histories extendin<sub>g</sub> ℎ.

Utilities. Advertisers have standard quasi-linear utilities. At a terminal histor<sub>y</sub> $h _ { T } ,$ advertiser �’s realized utilit<sub>y</sub> is $u _ { i } ^ { \Gamma } ( h _ { T } ; r _ { i } ) =$ $\begin{array} { r } { \lambda _ { i } ( h _ { T } ) r _ { i } ( \ell ( h _ { T } ) ) - \sum _ { t = 0 } ^ { T - 1 } \ d p _ { i } ( h _ { t } , m _ { t } , a _ { t } ) } \end{array}$ <sub>.</sub> F<sub>or</sub> <sub>a</sub> <sub>non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> histor<sub>y</sub> $h , \mathrm { ~ a ~ }$ t<sub>yp</sub>e <sub>p</sub>ro<sup>fil</sup>e $r , \mathrm { a }$ <sup>strate</sup>gy $\sigma _ { i : }$ , <sup>and</sup> <sup>o</sup>pp<sup>onent</sup> <sup>strate</sup>g<sup>ies</sup> $\sigma _ { - i }$ <sub>,</sub> the continuation ex<sub>p</sub>ected utilit<sub>y</sub> $U _ { i } ^ { \Gamma } ( h ; r _ { i } , \sigma _ { i } , \sigma _ { - i } ; r _ { - i } )$ i<sub>s</sub>

$$
\mathbb { E } _ { h _ { T } \sim \mathbb { P } _ { \sigma , r } ^ { \Gamma } ( \cdot | h ) } \left[ \lambda _ { i } ( h _ { T } ) r _ { i } ( \ell ( h _ { T } ) ) - \sum _ { t = t ( h ) } ^ { T - 1 } \mathfrak { p } _ { i } ( h _ { t } , m _ { t } , a _ { t } ) \right] .
$$

Truthful reporting. The truthful report is specified by a mechanisms<sub>p</sub>ecific t<sub>yp</sub>e-related re<sub>p</sub>ortin<sub>g</sub> rule $\eta _ { i } : \Theta _ { i } { \times } ( S \backslash S _ { T } )  \bigcup _ { h \in \mathcal { H } } \mathcal { M } _ { i } ( h )$ For exam<sub>p</sub>le<sub>,</sub> the sub-tree re<sub>p</sub>ortin<sub>g</sub> rule $\eta _ { i } ( r _ { i } , s ) = r _ { i } | _ { L ( s ) }$ <sub>an</sub>d th<sub>e</sub> value function re<sub>p</sub>ortin<sub>g</sub> rule $\eta _ { i } ( s ; r _ { i } ) = ( V _ { r _ { i } } ^ { * } ( [ s , a ] ) ) _ { a \in \mathcal { V } }$ <sub>.</sub> It d<sub>e</sub>fi<sub>nes</sub> th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> $\sigma _ { i } ^ { \mathrm { t r } } ( \boldsymbol { r } _ { i } , \boldsymbol { s } ) : = \eta _ { i } ( \boldsymbol { s } ; \boldsymbol { r } _ { i } )$

Because a se<sub>q</sub>uential mechanism allows advertisers to re<sub>p</sub>ort information d<sub>y</sub>namicall<sub>y,</sub> incentive com<sub>p</sub>atibilit<sub>y</sub> re<sub>q</sub>uires that an <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> <sub>canno</sub>t <sub>wa</sub>it f<sub>or</sub> <sub>a</sub> f<sub>avora</sub>bl<sub>e</sub> <sub>genera</sub>t<sub>e</sub>d <sub>pre</sub>fi<sub>x</sub> <sub>an</sub>d th<sub>en</sub> mani<sub>p</sub>ulate its future re<sub>p</sub>orts. Formall<sub>y,</sub>

Definition 1 (Markov DSIC and IR). The mechanism Γ is Markov dominant-strategy incentive compatible (Markov DSIC) if, for every advertiser �, type profile �, feasible opponent strategy pro-$\begin{array} { r l } {  { f i l e \sigma _ { - i } , } } \end{array}$ reachable platform history ℎ after advertiser � has reported truthfully along the prefix, and feasible continuation deviation $\sigma _ { i } ^ { \prime } ,$ $U _ { i } ^ { \Gamma } \left( h ; r _ { i } , \sigma _ { i } ^ { \mathrm { t r } } , \sigma _ { - i } ; r _ { - i } \right) \ge U _ { i } ^ { \Gamma } \left( h ; r _ { i } , \sigma _ { i } ^ { \prime } , \sigma _ { - i } ; r _ { - i } \right)$ . It is individually rational $( I R ) i f ,$ truthful participation from the initial state $s _ { 0 } = q$ gives nonnegative expected utility: $U _ { i } ^ { \Gamma } \left( q ; r _ { i } , \sigma _ { i } ^ { \mathrm { t r } } , \sigma _ { - i } ; r _ { - i } \right) \ge 0 \mathrm { ~ }$

KL-regularized social welfare. The platform’s objective is to generate native and useful content while allocatin<sub>g</sub> advertisin<sub>g</sub> o<sub>pp</sub>ortunities eficientl<sub>y</sub>. For a <sub>q</sub>uer<sub>y</sub> <sub>�,</sub> t<sub>yp</sub>e <sub>p</sub>rofile <sub>�,</sub> <sub>g</sub>eneration <sub>p</sub>olic<sub>y</sub> <sub>�,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> KL<sub>-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d <sub>soc</sub>i<sub>a</sub>l <sub>we</sub>lf<sub>are</sub> $\mathsf { S W } _ { \beta } ( \pi , \lambda \mid q , r )$ as

$$
\mathbb { E } _ { \tau \sim \pi ( \cdot | q ) } \left[ \sum _ { i \in N } \lambda _ { i } ( h _ { T } ) r _ { i } ( \ell ( \tau ) ) - \beta \sum _ { t = 0 } ^ { T _ { \tau } - 1 } \log \frac { \pi ( a _ { t } \mid s _ { t } ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } \right] .
$$

Th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm</sub> i<sub>s gross a</sub>d<sub>ver</sub>ti<sub>ser va</sub>l<sub>ue un</sub>d<sub>er</sub> th<sub>e genera</sub>t<sub>e</sub>d <sub>con</sub>t<sub>en</sub>t <sub>an</sub>d <sub>rea</sub>li<sub>ze</sub>d <sub>a</sub>ll<sub>oca</sub>ti<sub>on.</sub> Th<sub>e</sub> di<sub>vergence</sub> t<sub>erm pena</sub>li<sub>zes</sub> d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> from the or<sub>g</sub>anic <sub>p</sub>olic<sub>y,</sub> which o<sub>p</sub>timizes user ex<sub>p</sub>erience. <sup>4</sup> Thus $\beta > 0$ <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> th<sub>e</sub> b<sub>a</sub>l<sub>ance</sub> b<sub>e</sub>t<sub>ween mone</sub>ti<sub>za</sub>ti<sub>on an</sub>d <sub>con</sub>t<sub>en</sub>t <sub>na</sub>t<sub>ura</sub>l<sub>ness.</sub> Wh<sub>en</sub> th<sub>e query</sub> i<sub>s a</sub>l<sub>so ran</sub>d<sub>om,</sub> th<sub>e expec</sub>t<sub>e</sub>d <sub>we</sub>lf<sub>are</sub> i<sub>s</sub> $\mathbb { E } _ { q \sim \mathcal { D } } \left[ \operatorname { S W } _ { \beta } ( \pi , \lambda \mid q , r ) \right]$

Mechanism design problem. For each query-type pair $( q , r )$ , a se<sub>q</sub>uential mechanism Γ and its truthful re<sub>p</sub>ortin<sub>g</sub> rule induce a token <sub>p</sub>olic<sub>y</sub> $\pi _ { q , r } ^ { \Gamma }$ <sub>an</sub>d <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>ru</sub>l<sub>e</sub> $\lambda _ { q , r } ^ { \Gamma }$ f<sub>rom</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e.</sub> Th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>c</sub>h<sub>ooses</sub> th<sub>e</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> t<sub>o</sub> <sub>max</sub>i<sub>m</sub>i<sub>ze</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>we</sub>lf<sub>are:</sub>

$$
\operatorname* { m a x } _ { \Gamma } \quad \mathbb { E } _ { q \sim \mathcal { D } , r \sim F ( \cdot \vert q ) } \left[ \mathrm { S W } _ { \beta } \left( \pi _ { q , r } ^ { \Gamma } , \lambda _ { q , r } ^ { \Gamma } \mid q , r \right) \right]
$$

s.t. Γ is Markov DSIC and IR,

$$
\lambda _ { q , r } ^ { \Gamma } ( \tau ) \in \Lambda , \forall q , r , \tau .
$$

H<sub>e</sub>r<sub>e</sub> $F ( \cdot \mid q )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>con</sub>diti<sub>ona</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>over</sub> th<sub>e</sub> t<sub>ype</sub> <sub>pro</sub>fil<sub>es</sub> <sub>o</sub>f <sub>a</sub>d<sub>ver</sub>ti<sub>sers</sub> <sub>reca</sub>ll<sub>e</sub>d f<sub>or</sub> <sub>query</sub> $q .$ For ex-<sub>p</sub>ost anal<sub>y</sub>sis<sub>,</sub> the same o<sup>b</sup>jective can <sup>b</sup>e stu<sup>d</sup>ie<sup>d</sup> pointwise <sup>f</sup>or eac<sup>h</sup> rea<sup>l</sup>ize<sup>d</sup> $( q , r )$

## 3 Latent Advertiser Mixture Auction

We <sub>p</sub>ro<sub>p</sub>ose Latent Advertiser Mixture Auction (LAMA) to tackle this challen<sub>g</sub>e. The ke<sub>y</sub> idea is to view the <sub>g</sub>eneration <sub>p</sub>rocess as Ba<sub>y</sub>esian se<sub>q</sub>uential decision makin<sub>g</sub> that identifies the advertiser who values the <sub>g</sub>eneration o<sub>pp</sub>ortunit<sub>y</sub> most. As we will see<sub>,</sub> this <sub>perspec</sub>ti<sub>ve</sub> l<sub>ea</sub>d<sub>s</sub> t<sub>o a s</sub>i<sub>mp</sub>l<sub>e a</sub>l<sub>gor</sub>ith<sub>m</sub> th<sub>a</sub>t i<sub>s</sub> b<sub>o</sub>th f<sub>as</sub>t <sub>an</sub>d <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>.</sub>

Sequential reports. At a non-terminal state �, advertiser � reports <sub>a c</sub>hild<sub>-va</sub>l<sub>ue vec</sub>t<sub>or</sub> $\widehat { V } _ { i } ^ { s } = ( \widehat { V } _ { i } ^ { s } ( [ s , a ] ) ) _ { a \in \mathcal { V } }$ <sub>.</sub> Th<sub>e</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d t<sub>ru</sub>thf<sub>u</sub>l re<sub>p</sub>ort <sup>i</sup>s $\eta _ { i } ( s ; r _ { i } ) = \bigl ( V _ { i } ^ { * } ( [ s , a ] ) \bigr ) _ { a \in \mathcal { V } }$ <sub>, w</sub>h<sub>ere</sub> $V _ { i } ^ { * }$ i<sub>s</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>so</sub>ft <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y a</sub>d<sub>ver</sub>ti<sub>ser</sub> �’<sub>s</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d $r _ { i } { : }$

$$
V _ { i } ^ { * } ( s ) = \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \tau \sim \pi ( \cdot \vert s ) } \left[ r _ { i } ( \ell ( \tau ) ) - \beta \sum _ { t = 0 } ^ { T _ { \tau } - 1 } \log \frac { \pi ( a _ { t } \mid s _ { t } ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } \right] .
$$

Gi<sub>ven a repor</sub>t<sub>e</sub>d <sub>c</sub>hild<sub>-va</sub>l<sub>ue vec</sub>t<sub>or,</sub> d<sub>e</sub>fi<sub>ne a</sub>d<sub>ver</sub>ti<sub>ser</sub> $i ^ { \prime } s$ i<sub>n</sub>d<sub>uce</sub>d l<sub>oca</sub>l <sub>po</sub>li<sub>cy</sub> b<sub>y</sub>

$$
{ \widehat { \pi } } _ { i } ( a \mid s ; { \widehat { V } } _ { i } ^ { s } ) : = { \frac { \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \Big ( { \widehat { V } } _ { i } ^ { s } ( [ s , a ] ) / \beta \Big ) } { \sum _ { a ^ { \prime } \in { \mathcal { V } } } \pi _ { \mathrm { r e f } } ( a ^ { \prime } \mid s ) \exp \Big ( { \widehat { V } } _ { i } ^ { s } ( [ s , a ^ { \prime } ] ) / \beta \Big ) } } .
$$

Thi<sub>s</sub> i<sub>s</sub> th<sub>e nex</sub>t<sub>-</sub>t<sub>o</sub>k<sub>en po</sub>li<sub>cy</sub> th<sub>a</sub>t <sub>wou</sub>ld b<sub>e op</sub>ti<sub>ma</sub>l if <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> $i ^ { \prime } s$ <sub>repor</sub>t<sub>e</sub>d t<sub>ru</sub>thf<sub>u</sub>ll<sub>y.</sub>

Allocation posterior and next-token rule. The mechanism maintains an allocation <sub>p</sub>osterior $\rho ( s ) \in \Delta ( N )$ . <sup>A</sup>t t<sup>h</sup>e <sub>q</sub>uer<sub>y �</sub>, a<sup>f</sup>ter receivin<sub>g</sub> the first child-value re<sub>p</sub>orts<sub>,</sub> initialize $\begin{array} { r } { Z _ { i } ( q ) : = \sum _ { a \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid } \end{array}$ $q ) \exp ( \widehat { V } _ { i } ^ { q } ( [ q , a ] ) / \beta )$ <sub>, an</sub>d <sub>se</sub>t $\begin{array} { r } { \rho _ { i } ( q ) : = \frac { Z _ { i } ( q ) } { \sum _ { j \in N } Z _ { j } ( q ) } . } \end{array}$

At a reached non-terminal state <sub>�,</sub> after receivin<sub>g</sub> re<sub>p</sub>orts $\widehat { V } ^ { s } =$ $( \widehat { V } _ { 1 } ^ { s } , \ldots , \widehat { V } _ { n } ^ { s } )$ <sub>,</sub> th<sub>e</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> <sub>samp</sub>l<sub>es</sub> hi<sub>erarc</sub>hi<sub>ca</sub>ll<sub>y:</sub> fi<sub>rs</sub>t <sub>samp</sub>l<sub>e</sub> <sub>a</sub> l<sub>a</sub>t<sub>en</sub>t <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> $I \sim \rho ( s )$ <sub>an</sub>d th<sub>en genera</sub>t<sub>e</sub> th<sub>e nex</sub>t t<sub>o</sub>k<sub>en</sub> f<sub>rom</sub> th<sub>a</sub>t <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub>’<sub>s</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>po</sub>li<sub>cy,</sub> $a \sim \widehat \pi _ { I } ( \cdot \mid s ; \widehat V _ { I } ^ { s } )$ . Mar<sub>g</sub>inall<sub>y,</sub> this sam<sub>p</sub>lin<sub>g</sub> rule induces the mixture next-token distribution

$$
x ( a \mid s , \widehat V ^ { s } , \rho ( s ) ) : = \sum _ { i \in N } \rho _ { i } ( s ) \widehat \pi _ { i } ( a \mid s ; \widehat V _ { i } ^ { s } ) .
$$

The subse<sub>q</sub>uent <sub>p</sub>osterior u<sub>p</sub>date is <sub>p</sub>recisel<sub>y</sub> Ba<sub>y</sub>esian inference <sub>a</sub>b<sub>ou</sub>t thi<sub>s</sub> l<sub>a</sub>t<sub>en</sub>t <sub>a</sub>d<sub>ver</sub>ti<sub>ser a</sub>ft<sub>er o</sub>b<sub>serv</sub>i<sub>ng</sub> th<sub>e rea</sub>li<sub>ze</sub>d t<sub>o</sub>k<sub>en.</sub> Aft<sub>er</sub> t<sub>o</sub>k<sub>en �</sub> i<sub>s rea</sub>li<sub>ze</sub>d<sub>,</sub> th<sub>e pos</sub>t<sub>er</sub>i<sub>or</sub> i<sub>s up</sub>d<sub>a</sub>t<sub>e</sub>d b<sub>y</sub> B<sub>ayes</sub>’ <sub>ru</sub>l<sub>e:</sub>

$$
\rho _ { i } ( [ s , a ] ) = \frac { \rho _ { i } ( s ) \widehat \pi _ { i } ( a \mid s ; \widehat V _ { i } ^ { s } ) } { x ( a \mid s , \widehat V ^ { s } , \rho ( s ) ) } .
$$

Ifa terminal histor<sub>y</sub> $h _ { T }$ i<sub>s reac</sub>h<sub>e</sub>d<sub>,</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on ru</sub>l<sub>e</sub> i<sub>s</sub> $\lambda ( h _ { T } ) =$ $\rho ( \ell ( h _ { T } ) ) \in \Delta ( N )$

Payment rule. The payment rule has two components. First, after the initial re<sub>p</sub>orts at the <sub>q</sub>uer<sub>y</sub> <sub>p</sub>refix $q ,$ <sup>advertiser</sup> <sup>�</sup> p<sup>a</sup>y<sup>s</sup> <sup>an</sup> <sup>entr</sup>y f<sub>ee</sub> $p _ { i } ( q ; \widehat { V } )$ :

$$
\rho _ { i } ( q ) \widehat { V } _ { i } ( q ) - \beta \log \sum _ { j \in N } \exp ( \widehat { V } _ { j } ( q ) / \beta ) + \beta \log ( 1 + \sum _ { j \neq i } \exp ( \widehat { V } _ { j } ( q ) / \beta ) ) ,
$$

<sub>w</sub>h<sub>ere</sub> $\widehat V _ { i } ( q ) = \beta \log Z _ { i } ( q )$ i<sub>s</sub> th<sub>e roo</sub>t <sub>va</sub>l<sub>ue</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>c</sub>hild<sub>-va</sub>l<sub>ue</sub> <sub>repor</sub>t<sub>.</sub> S<sub>econ</sub>d<sub>,</sub> <sub>a</sub>l<sub>ong</sub> th<sub>e</sub> <sub>genera</sub>t<sub>e</sub>d <sub>pa</sub>th<sub>,</sub> th<sub>e</sub> <sub>mec</sub>h<sub>a-</sub> nism char<sub>g</sub>es advertisers accordin<sub>g</sub> to the chan<sub>g</sub>e in their <sub>p</sub>osteriorwei<sub>g</sub>hted re<sub>p</sub>orted contin<sub>u</sub>ation val<sub>u</sub>e. At a transition $s  [ s , a ]$ d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> i<sub>ns</sub>t<sub>an</sub>t<sub>aneous</sub> <sub>paymen</sub>t

$$
\begin{array} { r } { p _ { i } ( s , \widehat { V } ^ { s } , a ) : = \rho _ { i } ( [ s , a ] ) \widehat { V } _ { i } ^ { s } ( [ s , a ] ) - \rho _ { i } ( s ) \widehat { V } _ { i } ( s ) , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e paren</sub>t <sub>va</sub>l<sub>ue</sub> $\widehat { V } _ { i } ( s )$ i<sub>s</sub> th<sub>e va</sub>l<sub>ue prev</sub>i<sub>ous</sub>l<sub>y recor</sub>d<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> l<sub>e</sub>d<sub>ger.</sub> Th<sub>us,</sub> th<sub>e</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> <sub>c</sub>h<sub>arges</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> � <sub>w</sub>h<sub>en</sub> th<sub>e rea</sub>li<sub>ze</sub>d t<sub>o</sub>k<sub>en ma</sub>k<sub>es</sub> it<sub>s con</sub>ti<sub>nua</sub>ti<sub>on prospec</sub>t<sub>s more</sub> f<sub>avora</sub>bl<sub>e,</sub> <sub>a</sub>nd <sub>su</sub>b<sub>s</sub>idiz<sub>es</sub> it <sub>w</sub>h<sub>e</sub>n <sub>s</sub>it<sub>ua</sub>ti<sub>o</sub>n <sub>ge</sub>t<sub>s wo</sub>r<sub>se, p</sub>r<sub>ese</sub>r<sub>v</sub>in<sub>g</sub> it<sub>s</sub> in<sub>ce</sub>nti<sub>ve</sub> to continue <sub>p</sub>artici<sub>p</sub>atin<sub>g</sub> after an unfavorable transition.

Feasible message space. We require reports to satisfy the soft Bellman rec<sub>u</sub>rsion. This is eas<sub>y</sub> to verif<sub>y</sub> online: for each advertiser<sub>,</sub> th<sub>e p</sub>l<sub>a</sub>tf<sub>orm on</sub>l<sub>y nee</sub>d<sub>s</sub> t<sub>o s</sub>t<sub>ore a sca</sub>l<sub>ar,</sub> i<sub>,e.</sub> th<sub>e va</sub>l<sub>ue prev</sub>i<sub>ous</sub>l<sub>y</sub> <sub>repor</sub>t<sub>e</sub>d f<sub>or</sub> th<sub>e</sub> <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> <sub>an</sub>d <sub>compare</sub> it <sub>w</sub>ith th<sub>e</sub> <sub>so</sub>ft B<sub>e</sub>ll<sub>man</sub> <sub>aggrega</sub>ti<sub>on o</sub>f th<sub>e new</sub>l<sub>y repor</sub>t<sub>e</sub>d <sub>c</sub>hild <sub>va</sub>l<sub>ues.</sub> F<sub>orma</sub>ll<sub>y,</sub>

Definition 2 (Bellman consistency). At a history ℎ with current state $s = s ( h )$ , let $\widehat { V } _ { i } ( s )$ be advertiser �’s value for � already stored in the ledger. The feasible message space is

$$
\begin{array} { r } { M _ { i } ( h ) : = \left\{ \begin{array} { l l } { \mathbb { R } ^ { | \mathcal { V } | } , } & { s = q , } \\ { \{ v \in \mathbb { R } ^ { | \mathcal { V } | } : \beta \log \sum _ { a \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid s ) e ^ { v _ { a } / \beta } = \widehat { V } _ { i } ( s ) \} , } & { s \neq q . } \end{array} \right. } \end{array}
$$

Here $v _ { a }$ is the reported value ofthe child state [�, �] in $\widehat { V } _ { i } ^ { s }$

We now read<sub>y</sub> to show the main incentive <sub>g</sub>uarantee.

Theorem 1 (Markov DSIC and IR of LAMA). The LAMA with Bellman-consistent message spaces is Markov DSIC and $I R f o r$ the truthful reporting rule $\eta _ { i } ( s ; r _ { i } ) = \bigl ( V _ { i } ^ { * } ( [ s , a ] ) \bigr ) _ { a \in \mathcal { V } }$

Proof sketch. First consider <sub>g</sub>loball<sub>y</sub> Bellman-consistent direct re<sub>p</sub>orts and write $z _ { i } ( s ) = \exp ( \widehat { V } _ { i } ( s ) / \beta )$ <sub>an</sub>d $\begin{array} { r } { \overline { { z } } ( s ) = \sum _ { j } z _ { j } ( s ) } \end{array}$ B<sub>e</sub>ll<sub>man cons</sub>i<sub>s</sub>t<sub>ency ma</sub>k<sub>es eac</sub>h $z _ { i }$ h<sub>armon</sub>i<sub>c un</sub>d<sub>er</sub> $\pi _ { \mathrm { r e f } }$ <sub>.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>tin<sub>g</sub> this identit<sub>y</sub> into the Ba<sub>y</sub>esian u<sub>p</sub>date <sub>y</sub>ields $\rho _ { i } ( s ) = z _ { i } ( s ) / \overline { { z } } ( s )$ <sub>an</sub>d $x ( a \mid s ) = \pi _ { \mathrm { r e f } } ( a \mid s ) { \overline { { z } } } ( [ s , a ] ) / { \overline { { z } } } ( s )$ <sub>.</sub> Th<sub>e</sub> f<sub>ac</sub>t<sub>ors</sub> t<sub>e</sub>l<sub>escope a</sub>l<sub>ong</sub> a pat<sup>h</sup>, so t<sup>h</sup>e joint pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity t<sup>h</sup>at termina<sup>l</sup> state ℓ is generate<sup>d</sup> <sub>an</sub>d <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub>d t<sub>o a</sub>d<sub>ver</sub>ti<sub>ser</sub> � i<sub>s</sub>

$$
\alpha _ { i , \ell } ( \widehat { V } ) = \frac { P _ { 0 } ( \ell \mid q ) \exp ( \widehat { V } _ { i } ( \ell ) / \beta ) } { \sum _ { \ell ^ { \prime } } P _ { 0 } ( \ell ^ { \prime } \mid q ) \sum _ { j } \exp ( \widehat { V } _ { j } ( \ell ^ { \prime } ) / \beta ) } .
$$

This is exactl<sub>y</sub> the <sub>g</sub>radient<sub>,</sub> with res<sub>p</sub>ect to terminal re<sub>p</sub>orts<sub>,</sub> of the convex lo<sub>g</sub>-sum-ex<sub>p</sub> <sub>p</sub>otential $\Phi _ { q }$ d<sub>e</sub>fi<sub>ne</sub>d i<sub>n</sub> th<sub>e</sub> <sub>appen</sub>di<sub>x.</sub> Th<sub>e</sub> telesco<sub>p</sub>in<sub>g</sub> LAMA <sub>p</sub>a<sub>y</sub>ment is the corres<sub>p</sub>ondin<sub>g</sub> <sub>p</sub>otential-based p<sup>a</sup>y<sup>ment</sup>, <sup>so</sup> <sup>the</sup> <sup>convex</sup> <sup>su</sup>pp<sup>ortin</sup>g <sup>h</sup>yp<sup>er</sup>p<sup>lane</sup> <sup>ine</sup>q<sup>ualit</sup>y g<sup>ives</sup> direct truthfulness. Nonne<sub>g</sub>ative rewards im<sub>p</sub>l<sub>y</sub> $z _ { i } ^ { * } ( q ) \geq 1$ <sub>, w</sub>hi<sub>c</sub>h makes truthful utilit<sub>y</sub> nonne<sub>g</sub>ative relative to the outside o<sub>p</sub>tion.

It r<sub>e</sub>m<sub>a</sub>in<sub>s</sub> t<sub>o co</sub>nn<sub>ec</sub>t <sub>o</sub>nlin<sub>e</sub> r<sub>epo</sub>rt<sub>s</sub> t<sub>o</sub> thi<sub>s</sub> dir<sub>ec</sub>t <sub>co</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n<sub>.</sub> Fi<sub>x a reac</sub>h<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e � a</sub>ft<sub>er a</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>pre</sub>fi<sub>x.</sub> A<sub>ny</sub> f<sub>eas</sub>ibl<sub>e con</sub>ti<sub>nua-</sub> ti<sub>on</sub> d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> d<sub>e</sub>fi<sub>nes a</sub> B<sub>e</sub>ll<sub>man-cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on</sub> i<sub>ns</sub>id<sub>e</sub> th<sub>e su</sub>bt<sub>ree roo</sub>t<sub>e</sub>d <sub>a</sub>t <sub>�.</sub> Wh<sub>en �</sub> $; \neq q ,$ <sub>,</sub> f<sub>eas</sub>ibilit<sub>y</sub> f<sub>orces</sub> it<sub>s roo</sub>t <sub>va</sub>l<sub>ue</sub> t<sub>o equa</sub>l th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>va</sub>l<sub>ue a</sub>l<sub>rea</sub>d<sub>y s</sub>t<sub>ore</sub>d <sub>a</sub>t <sub>�; w</sub>h<sub>en</sub> $s = q ,$ th<sub>ere are</sub> no ancestors to <sub>p</sub>reserve. Use the deviatin<sub>g</sub> values inside the subtree <sub>an</sub>d $V _ { i } ^ { * }$ <sub>everyw</sub>h<sub>ere e</sub>l<sub>se.</sub> Th<sub>e resu</sub>lt i<sub>s a g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> B<sub>e</sub>ll<sub>man-cons</sub>i<sub>s</sub>t<sub>en</sub>t direct re<sub>p</sub>ort with the same continuation <sub>p</sub>olic<sub>y,</sub> allocation<sub>,</sub> and <sub>p</sub>a<sub>y</sub> m<sub>e</sub>nt<sub>s as</sub> th<sub>e o</sub>nlin<sub>e</sub> d<sub>ev</sub>i<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> Dir<sub>ec</sub>t DSIC <sub>a</sub>nd IR th<sub>e</sub>n imm<sub>e</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> im<sub>p</sub>l<sub>y</sub> Markov DSIC and IR for the se<sub>qu</sub>ential mechanism. □

Eficiency. LAMA also provides strong eficiency guarantee. In tuitivel<sub>y,</sub> it im<sub>p</sub>lements two soft lotteries: one over <sub>g</sub>enerated res<sub>p</sub>onses throu<sub>g</sub>h KL-re<sub>g</sub>ularized decodin<sub>g</sub> and one over advertisers th<sub>roug</sub>h <sub>en</sub>t<sub>ropy-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d <sub>a</sub>ll<sub>oca</sub>ti<sub>on.</sub> Th<sub>e resu</sub>lti<sub>ng we</sub>lf<sub>are</sub> l<sub>oss</sub> i<sub>s</sub> at most � lo<sub>g</sub> |N| relative to the o<sub>p</sub>timal KL-re<sub>g</sub>ularized benchmark, <sub>an</sub>d th<sub>e mec</sub>h<sub>an</sub>i<sub>sm approac</sub>h<sub>es</sub> fi<sub>rs</sub>t<sub>-</sub>b<sub>es</sub>t <sub>we</sub>lf<sub>are as</sub> $\beta \to 0 ^ { + }$

Theorem 2 (Optimization objective and asymptotic efficiency). Fix a query � and a type profile �. Let $( x ^ { \beta } , \lambda ^ { \beta } )$ be the truthful policy and terminal allocation induced by LAMA. Then $( x ^ { \beta } , \lambda ^ { \beta } )$ maximizes the entropy-relaxed welfare:

$$
\mathbb { E } _ { \tau \sim x ( \cdot \vert q ) } \left[ \sum _ { i \in N } \lambda _ { i } ( \tau ) r _ { i } ( \ell ( \tau ) ) + \beta \mathcal H ( \lambda ( \tau ) ) - \beta \sum _ { t = 0 } ^ { T _ { \tau } - 1 } \log \frac { x ( a _ { t } \mid s _ { t } ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } \right] .
$$

Moreover, relative to the optimal KL-regularized welfare in the simplex allocation setting, $\begin{array} { r } { \operatorname* { s u p } _ { \pi , \lambda } \mathrm { S W } _ { \beta } ( \pi , \lambda \mid q , r ) , } \end{array}$ , the welfare loss of the Latent Advertiser Mixture mechanism is bounded by

$$
0 \leq \operatorname* { s u p } _ { \pi , \lambda } \mathrm { S W } _ { \beta } ( \pi , \lambda \mid q , r ) - \mathrm { S W } _ { \beta } ( x ^ { \beta } , \lambda ^ { \beta } \mid q , r ) \leq \beta \log | N | .
$$

Thus, lim ${ } _ { \beta \to 0 ^ { + } } \operatorname { S W } _ { \beta } ( x ^ { \beta } , \lambda ^ { \beta } \mid q , r ) = \operatorname * { m a x } _ { \ell \in L ( q ) } \operatorname * { m a x } _ { i \in N } r _ { i } ( \ell )$ . The same welfare-gap bound holds after taking expectation over $q \sim \mathcal { D }$

## 4 Practical Implementation

In <sub>p</sub>ractice<sub>,</sub> com<sub>p</sub>utin<sub>g</sub> the re<sub>p</sub>orts re<sub>q</sub>uired b<sub>y</sub> LAMA is <sub>g</sub>ener-<sub>a</sub>ll<sub>y</sub> b<sub>eyon</sub>d th<sub>e capa</sub>biliti<sub>es o</sub>f i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>a</sub>d<sub>ver</sub>ti<sub>sers.</sub> At <sub>eac</sub>h <sub>non-</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e �, a</sub>d<sub>ver</sub>ti<sub>ser</sub> � <sub>wou</sub>ld <sub>nee</sub>d t<sub>o su</sub>b<sub>m</sub>it th<sub>e con</sub>ti<sub>nua</sub>ti<sub>on</sub> vector $\left( V _ { i } ^ { * } ( \left[ s , a \right] ) \right) _ { a \in \mathcal { V } }$ <sub>.</sub> Doin<sub>g</sub> so re<sub>q</sub>uires both hi<sub>g</sub>h-<sub>q</sub>ualit<sub>y</sub> user f<sub>ee</sub>db<sub>ac</sub>k d<sub>a</sub>t<sub>a an</sub>d <sub>cons</sub>id<sub>era</sub>bl<sub>e compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>resources.</sub> W<sub>e</sub> th<sub>ere-</sub> fore <sub>p</sub>rovide re<sub>p</sub>ortin<sub>g</sub> as a <sub>p</sub>latform-side service analo<sub>g</sub>ous to autobiddin<sub>g</sub> [1, 17]: the <sub>p</sub>latform learns value re<sub>p</sub>orts from advertisin<sub>g</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>an</sub>d <sub>serves</sub> th<sub>em on a</sub>d<sub>ver</sub>ti<sub>sers</sub>’ b<sub>e</sub>h<sub>a</sub>lf<sub>.</sub>

A<sub>s we w</sub>ill <sub>s</sub>h<sub>ow,</sub> th<sub>e p</sub>l<sub>a</sub>tf<sub>orm can avo</sub>id l<sub>earn</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>so</sub>ft value function throu<sub>g</sub>h reinforcement learnin<sub>g</sub> b<sub>y</sub> decom<sub>p</sub>osin<sub>g</sub> re<sub>p</sub>ortin<sub>g</sub> into two sim<sub>p</sub>ler sta<sub>g</sub>es. It learns local soft advanta<sub>g</sub>es<sub>,</sub> w<sup>h</sup>ic<sup>h</sup> <sup>d</sup>etermine c<sup>h</sup>anges in va<sup>l</sup>ue <sup>b</sup>etween a<sup>d</sup>jacent pre<sup>fi</sup>xes, an<sup>d</sup> <sub>a</sub> <sub>roo</sub>t <sub>va</sub>l<sub>ue,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>anc</sub>h<sub>ors</sub> th<sub>e</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>sca</sub>l<sub>e.</sub> Th<sub>ese</sub> t<sub>wo</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>s</sub> <sub>are</sub> th<sub>en assem</sub>bl<sub>e</sub>d i<sub>n</sub>t<sub>o</sub> th<sub>e requ</sub>i<sub>re</sub>d <sub>repor</sub>t <sub>a</sub>l<sub>ong</sub> th<sub>e rea</sub>li<sub>ze</sub>d <sub>pa</sub>th<sub>.</sub> B<sub>o</sub>th <sub>s</sub>t<sub>ages</sub> <sub>can</sub> <sub>use</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>superv</sub>i<sub>se</sub>d <sub>s</sub>i<sub>gna</sub>l<sub>s</sub> <sub>an</sub>d b<sub>e</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>e</sub>d b<sub>y a</sub>d<sub>ver</sub>ti<sub>ser-con</sub>diti<sub>one</sub>d <sub>serv</sub>i<sub>ces s</sub>h<sub>are</sub>d <sub>across a</sub>d<sub>ver</sub>ti<sub>sers.</sub>

## 4.1 Value Decomposition

F<sub>or a pre</sub>fi<sub>x � an</sub>d <sub>nex</sub>t t<sub>o</sub>k<sub>en �,</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>so</sub>ft <sub>a</sub>d<sub>van</sub>t<sub>age</sub> f<sub>or</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> � <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $A _ { i } ^ { * } ( s , a ) = V _ { i } ^ { * } ( [ s , a ] ) - V _ { i } ^ { * } ( s ) ^ { 5 }$ <sub>.</sub> F<sub>or a comp</sub>l<sub>e</sub>t<sub>e</sub>d <sup>res</sup>p<sup>onse</sup> $y = \left( a _ { 0 } , \ldots , a _ { T _ { y } - 1 } \right) $ <sub>,</sub> l<sub>e</sub>t $s _ { t } = [ q , a _ { < t } ] , \ell _ { y } = [ q , y ]$ <sub>.</sub> Th<sub>en</sub> th<sub>e va</sub>l<sub>ue o</sub>f <sub>every pre</sub>fi<sub>x can</sub> b<sub>e o</sub>bt<sub>a</sub>i<sub>ne</sub>d f<sub>rom</sub> th<sub>e roo</sub>t <sub>va</sub>l<sub>ue</sub> b<sub>y</sub> te<sup>l</sup>esco<sub>p</sub><sup>i</sup>n<sub>g</sub>: $\begin{array} { r } { V _ { i } ^ { * } ( s _ { t } ) = V _ { i } ^ { * } ( q ) + \sum _ { k = 0 } ^ { t - 1 } A _ { i } ^ { * } ( s _ { k } , a _ { k } ) } \end{array}$ <sub>.</sub> Th<sub>us,</sub> t<sub>o</sub> <sub>cons</sub>t<sub>ruc</sub>t th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>c</sub>hild<sub>-va</sub>l<sub>ue repor</sub>t <sub>a</sub>t <sub>any reac</sub>h<sub>a</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>e,</sub> it i<sub>s enoug</sub>h t<sub>o</sub> k<sub>now</sub> th<sub>e roo</sub>t <sub>va</sub>l<sub>ue</sub> $V _ { i } ^ { * } ( q )$ <sub>an</sub>d th<sub>e</sub> l<sub>oca</sub>l <sub>a</sub>d<sub>van</sub>t<sub>ages</sub> $A _ { i } ^ { * } ( s , a )$

We now present t<sup>h</sup>e <sup>l</sup>earning o<sup>b</sup>jectives <sup>f</sup>or t<sup>h</sup>ese two <sup>f</sup>unctions. D<sub>e</sub>fi<sub>ne</sub> $r _ { i } ( \ell _ { y } )$ as advertiser �’s <sub>p</sub>o<sub>p</sub>ulation value for receivin<sub>g</sub> the advertisin<sub>g</sub> im<sub>p</sub>ression o<sub>pp</sub>ortunit<sub>y</sub> from res<sub>p</sub>onse <sub>�</sub>. Let D denote the <sub>p</sub>o<sub>p</sub>ulation com<sub>p</sub>arison distribution over $( i , q , y , y ^ { \prime } )$ <sub>, an</sub>d <sub>use</sub> $\mathcal { D } ( \cdot \mid i , q )$ for its sin<sub>g</sub>le-res<sub>p</sub>onse mar<sub>g</sub>inal when onl<sub>y</sub> <sub>�</sub> is needed.

I. Local advantages. For any policy $\pi _ { i } ,$ d<sub>e</sub>fi<sub>ne</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve re-</sub> <sup>s</sup>p<sup>onse</sup> <sup>score</sup> <sup>as</sup> $\begin{array} { r } { \bar { G _ { i } ^ { \pi } } ( q , y ) : = \beta \log \frac { \pi _ { i } ( y \vert q ) } { \pi _ { \mathrm { r e f } } ( y \vert q ) } = \beta \sum _ { t = 0 } ^ { T _ { y } - 1 } \log \frac { \pi _ { i } ( a _ { t } \vert s _ { t } ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \vert s _ { t } ) } . } \end{array}$ T<sub>o</sub> l<sub>earn</sub> th<sub>ese</sub> t<sub>o</sub>k<sub>en-</sub>l<sub>eve</sub>l l<sub>og-ra</sub>ti<sub>os,</sub> th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>samp</sub>l<sub>es</sub> <sub>pa</sub>i<sub>re</sub>d <sup>res</sup>p<sup>onses</sup> � <sup>and</sup> $y ^ { \prime }$ under the same advertiser-<sub>q</sub>uer<sub>y</sub> <sub>p</sub>air $( i , q )$ <sub>.</sub> Fr<sub>o</sub>m their scalar rewards, it constructs the Bradle<sub>y</sub>-Terr<sub>y</sub> [5] win rate

$$
\omega _ { i } ( q ; y , y ^ { \prime } ) : = \operatorname* { P r } ( y \succ y ^ { \prime } \mid i , q ) = \sigma \big ( r _ { i } ( \ell _ { y } ) - r _ { i } ( \ell _ { y ^ { \prime } } ) \big ) .
$$

Let $\Delta _ { i } ^ { \pi } ( q ; y , y ^ { \prime } ) : = G _ { i } ^ { \pi } ( q , y ) - G _ { i } ^ { \pi } ( q , y ^ { \prime } )$ <sub>.</sub> Th<sub>e</sub> id<sub>ea</sub>l l<sub>oca</sub>l<sub>-a</sub>d<sub>van</sub>t<sub>age</sub> o<sup>b</sup>jective is t<sup>h</sup>ere<sup>f</sup>ore t<sup>h</sup>e <sup>b</sup>inary cross-entropy,

$$
\begin{array} { r l } & { \mathcal { L } ( \pi ) : = - \mathbb { E } _ { ( i , q , y , y ^ { \prime } ) \sim \mathcal { D } } \Big [ \omega _ { i } ( q ; y , y ^ { \prime } ) \log \sigma \big ( \Delta _ { i } ^ { \pi } ( q ; y , y ^ { \prime } ) \big ) } \\ & { \qquad + \left. \left( 1 - \omega _ { i } ( q ; y , y ^ { \prime } ) \right) \log \sigma \big ( - \Delta _ { i } ^ { \pi } ( q ; y , y ^ { \prime } ) \big ) \ \right] . } \end{array}
$$

A<sub>s we w</sub>ill <sub>s</sub>h<sub>ow</sub> l<sub>a</sub>t<sub>er, un</sub>d<sub>er</sub> th<sub>e usua</sub>l <sub>regu</sub>l<sub>ar</sub>it<sub>y con</sub>diti<sub>ons,</sub> th<sub>e pop-</sub> <sub>u</sub>l<sub>a</sub>ti<sub>o</sub>n minimiz<sub>e</sub>r $\pi ^ { \dagger }$ <sub>ma</sub>t<sub>c</sub>h<sub>es</sub> th<sub>e</sub> B<sub>ra</sub>dl<sub>ey-</sub>T<sub>erry</sub> l<sub>og-o</sub>dd<sub>s</sub> <sub>an</sub>d th<sub>ere-</sub> f<sub>ore</sub> id<sub>en</sub>tifi<sub>es rewar</sub>d dif<sub>erences</sub> b<sub>e</sub>t<sub>ween comp</sub>l<sub>e</sub>t<sub>e</sub>d <sub>responses</sub> <sub>un</sub>d<sub>er</sub> th<sub>e same</sub> $( i , q )$ <sub>.</sub> I<sub>n</sub> th<sub>e</sub> <sub>so</sub>ft<sub>-con</sub>t<sub>ro</sub>l <sub>v</sub>i<sub>ew,</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> l<sub>oca</sub>l l<sub>og-ra</sub>ti<sub>o</sub> $\widehat A _ { i } ^ { \dagger } ( s , a ) : = \beta \log \pi _ { i } ^ { \dagger } ( a \mid s ) / \pi _ { \mathrm { r e f } } ( a \mid s )$ <sub>recovers</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>so</sub>ft <sub>a</sub>d<sub>van</sub>t<sub>age</sub> $A _ { i } ^ { * } ( s , a )$

II. Root values. The root value is anchored separately from scalar <sub>rewar</sub>d<sub>s.</sub> Si<sub>nce a</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> h<sub>as va</sub>l<sub>ue</sub> $V _ { i } ^ { * } ( \ell _ { y } ) = r _ { i } ( \ell _ { y } )$ , te<sup>l</sup>esco<sub>p</sub>- <sup>i</sup>n<sub>g</sub> <sub>g</sub><sup>i</sup>ves, <sup>f</sup>or ever<sub>y</sub> com<sub>p</sub><sup>l</sup>ete<sup>d</sup> res<sub>p</sub>onse,

$$
V _ { i } ^ { * } ( q ) = r _ { i } ( \ell _ { y } ) - \sum _ { t = 0 } ^ { T _ { y } - 1 } A _ { i } ^ { * } ( s _ { t } , a _ { t } ) .
$$

Avera<sub>g</sub>in<sub>g</sub> this residual over <sub>p</sub>o<sub>p</sub>ulation res<sub>p</sub>onses for the same $( i , q )$ <sub>recovers</sub> th<sub>e</sub> <sub>roo</sub>t <sub>va</sub>l<sub>ue.</sub> Th<sub>e</sub> t<sub>wo</sub> id<sub>ea</sub>l l<sub>earn</sub>i<sub>ng</sub> t<sub>as</sub>k<sub>s</sub> <sub>are</sub> th<sub>ere</sub>f<sub>ore:</sub> l<sub>earn</sub> l<sub>oca</sub>l <sub>a</sub>d<sub>van</sub>t<sub>ages</sub> f<sub>rom</sub> BT<sub>-we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> dif<sub>erences, an</sub>d th<sub>ere</sub>b<sub>y anc</sub>h<sub>or</sub> th<sub>e roo</sub>t <sub>va</sub>l<sub>ue w</sub>ith t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d<sub>s.</sub>

Th<sub>e nex</sub>t th<sub>eorem s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e popu</sub>l<sub>a</sub>ti<sub>on</sub> l<sub>oca</sub>l<sub>-a</sub>d<sub>van</sub>t<sub>age</sub> <sub>op</sub>ti<sub>mum</sub> id<sub>en</sub>tifi<sub>es</sub> b<sub>o</sub>th th<sub>e</sub> l<sub>oca</sub>l <sub>so</sub>ft <sub>a</sub>d<sub>van</sub>t<sub>ages</sub> <sub>an</sub>d <sub>a</sub> <sub>res</sub>id<sub>ua</sub>l f<sub>ormu</sub>l<sub>a</sub> f<sub>or</sub> th<sub>e roo</sub>t <sub>va</sub>l<sub>ue.</sub> Th<sub>e</sub> l<sub>e</sub>d<sub>ger recurs</sub>i<sub>on</sub> i<sub>s</sub> th<sub>en ena</sub>bl<sub>e</sub>d t<sub>o</sub> <sub>recons</sub>t<sub>ruc</sub>t th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>so</sub>ft <sub>va</sub>l<sub>ue</sub> <sub>on</sub> <sub>every</sub> <sub>reac</sub>h<sub>e</sub>d <sub>pre</sub>fi<sub>x,</sub> f<sub>orma</sub>ll<sub>y</sub> justi<sup>f</sup>ying t<sup>h</sup>e va<sup>l</sup>ue <sup>d</sup>ecomposition a<sup>b</sup>ove.

![](images/f83721cd49c73ba0e88d32f37da13a3f2403eb6616bcff8fec056d8a96bd3c4f.jpg)  
Figure 2: Overview of the Latent Advertiser Mixture Auction (LAMA). (1) Report: Advertisers report soft values for the current child states, and the platform verifies Bellman consistency against the value stored in its ledger. (2) Next-token generation: The platform samples an advertiser from its current belief, whose optimal policy generates the next token. (3) Update: After the context changes, the platform updates the ledger and its belief over the optimal advertiser via Bayes’ rule. (4) Settlement: Once the response is complete, the platform samples a winning advertiser from the posterior; the winner pays the corresponding charge and receives the hyperlink placement. See Figure 3 for additional impression formats.

Theorem 3 (Exact recovery). Suppose D has full comparison support<sup>6</sup>. Then the optimal token policy $\pi ^ { * }$ is the unique minimizer ofthe local-advantage objective L. Relatedly, the recovered log-ratio $\begin{array} { r } { \widehat { A } _ { i } ^ { \dagger } ( s , a ) : = \beta \log \pi _ { i } ^ { \dagger } ( a \mid s ) / \pi _ { \mathrm { r e f } } ( a \mid s ) } \end{array}$ equals $A _ { i } ^ { * } ( s , a )$ . Also, the root value $\begin{array} { r } { v _ { i } ^ { \dagger } ( q ) : = \mathbb { E } _ { y \sim \mathcal { D } ( \cdot | i , q ) } \left[ r _ { i } ( \ell _ { y } ) - \sum _ { t = 0 } ^ { T _ { y } - 1 } \widehat { A } _ { i } ^ { \dagger } ( s _ { t } , a _ { t } ) \right] } \end{array}$ equals $V _ { i } ^ { * } ( q )$ Finally, initialize $\widehat { V } _ { i } ( q ) : = v _ { i } ^ { \dagger } ( q )$ andrecursively define $\widehat { V } _ { i } ( [ s , a ] ) : =$ $\widehat { V } _ { i } ( s ) + \widehat { A } _ { i } ^ { \dagger } ( s , a )$ . Then $\widehat { V } _ { i } ( s ) = V _ { i } ^ { * } ( s )$ at every reachable prefix. Consequently, the resulting child-value vector is exactly the truthful softvalue report required by LAMA.

Shared report model. Motivated by this recovery result, and to <sub>re</sub>d<sub>uce</sub> th<sub>e cos</sub>t <sub>o</sub>f t<sub>ra</sub>i<sub>n</sub>i<sub>ng an</sub>d <sub>serv</sub>i<sub>ng separa</sub>t<sub>e mo</sub>d<sub>e</sub>l<sub>s,</sub> th<sub>e p</sub>l<sub>a</sub>t f<sub>orm</sub> i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>es</sub> th<sub>e</sub> t<sub>wo</sub> l<sub>earne</sub>d <sub>componen</sub>t<sub>s w</sub>ith <sub>a s</sub>i<sub>ng</sub>l<sub>e s</sub>h<sub>are</sub>d <sub>pre</sub>t<sub>ra</sub>i<sub>ne</sub>d b<sub>ac</sub>kb<sub>one.</sub> All <sub>a</sub>d<sub>ver</sub>ti<sub>sers s</sub>h<sub>are</sub> th<sub>e parame</sub>t<sub>ers, w</sub>hil<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>spec</sub>ifi<sub>c</sub> t<sub>o</sub> <sub>eac</sub>h <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> � <sub>en</sub>t<sub>ers</sub> <sub>on</sub>l<sub>y</sub> th<sub>roug</sub>h <sub>a</sub> <sub>con-</sub> ditionin<sub>g</sub> <sub>p</sub>rom<sub>p</sub>t $\kappa _ { i } ,$ <sub>w</sub>hich contains its identit<sub>y</sub> and cam<sub>p</sub>ai<sub>g</sub>n information<sub>,</sub> such as the creative<sub>,</sub> landin<sub>g p</sub>a<sub>g</sub>e<sub>,</sub> and tar<sub>g</sub>etin<sub>g</sub> context. For an<sub>y</sub> <sub>g</sub>eneration <sub>p</sub>refix $s ,$ th<sub>e</sub> <sub>same</sub> <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> <sub>run</sub> <sub>on</sub> th<sub>e</sub> <sub>conca</sub>t<sub>e-</sub> <sub>na</sub>t<sub>e</sub>d <sub>con</sub>t<sub>ex</sub>t $[ \kappa _ { i } , s ]$ and <sub>p</sub>roduces next-token lo<sub>g</sub>its �<sub>�</sub> (� | [�<sub>�</sub>, �]). Th<sub>e</sub> <sub>po</sub>li<sub>cy</sub> f<sub>or</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> � i<sub>s</sub> th<sub>e</sub> <sub>so</sub>ft<sub>max</sub> <sub>o</sub>f th<sub>ese</sub> l<sub>og</sub>it<sub>s:</sub>

$$
\pi _ { \theta , i } ( a \mid s ) : = { \mathrm { s o f t m a x } } _ { a } z _ { \theta } ( a \mid [ \kappa _ { i } , s ] ) = { \frac { \exp z _ { \theta } ( a \mid [ \kappa _ { i } , s ] ) } { \sum _ { b \in \mathcal { N } } \exp z _ { \theta } ( b \mid [ \kappa _ { i } , s ] ) } } .
$$

A<sub>n</sub>d th<sub>e roo</sub>t <sub>anc</sub>h<sub>or</sub> i<sub>s pro</sub>d<sub>uce</sub>d b<sub>y a</sub> fi<sub>ne-</sub>t<sub>une</sub>d <sub>sca</sub>l<sub>ar va</sub>l<sub>ue</sub> h<sub>ea</sub>d <sub>a</sub>tt<sub>ac</sub>h<sub>e</sub>d t<sub>o</sub> th<sub>e same s</sub>h<sub>are</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>on:</sub>

$$
v _ { \phi , i } ( q ) : = v _ { \phi } ( [ \kappa _ { i } , q ] ) .
$$

T<sup>h</sup>e <sup>d</sup>ep<sup>l</sup>oye<sup>d</sup> o<sup>b</sup>ject is t<sup>h</sup>ere<sup>f</sup>ore a sing<sup>l</sup>e s<sup>h</sup>are<sup>d</sup> mo<sup>d</sup>e<sup>l</sup> t<sup>h</sup>at pro<sup>d</sup>uces <sub>a</sub>d<sub>ver</sub>ti<sub>ser-spec</sub>ifi<sub>c</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>s:</sub> <sub>nex</sub>t<sub>-</sub>t<sub>o</sub>k<sub>en</sub> l<sub>og</sub>it<sub>s</sub> f<sub>or</sub> l<sub>oca</sub>l <sub>repor</sub>t<sub>s</sub> <sub>an</sub>d a scalar root value for initializin<sub>g</sub> the led<sub>g</sub>er. A<sub>pp</sub>endix A <sub>g</sub>ives the <sub>p</sub>ractical finite-sam<sub>p</sub>le trainin<sub>g</sub> <sub>p</sub>rocedure in Al<sub>g</sub>orithm 2.

## 4.2 Serving-time Procedure

Fi<sub>gu</sub>r<sub>e</sub> 2 <sub>p</sub>r<sub>ov</sub>id<sub>es a</sub>n int<sub>u</sub>iti<sub>ve ove</sub>r<sub>v</sub>i<sub>ew o</sub>f LAMA<sub>.</sub> At <sub>se</sub>r<sub>v</sub>in<sub>g</sub> tim<sub>e,</sub> t<sup>h</sup>e <sup>l</sup>earne<sup>d</sup> o<sup>b</sup>jects a<sup>b</sup>ove are expose<sup>d</sup> as a sing<sup>l</sup>e report service. F<sub>or eac</sub>h <sub>query �,</sub> th<sub>e p</sub>l<sub>a</sub>tf<sub>orm</sub> fi<sub>rs</sub>t <sub>re</sub>t<sub>r</sub>i<sub>eves a sma</sub>ll <sub>can</sub>did<sub>a</sub>t<sub>e se</sub>t <sub>o</sub>f <sub>a</sub>d<sub>ver</sub>ti<sub>sers</sub> th<sub>roug</sub>h th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng</sub> f<sub>unne</sub>l<sub>.</sub> F<sub>or eac</sub>h <sub>re</sub>t<sub>r</sub>i<sub>eve</sub>d <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> �<sub>,</sub> it b<sub>u</sub>ild<sub>s</sub> th<sub>e a</sub>d<sub>ver</sub>ti<sub>ser pre</sub>fi<sub>x</sub> $\kappa _ { i } ,$ <sub>eva</sub>l<sub>ua</sub>t<sub>es</sub> th<sub>e</sub> <sub>roo</sub>t <sub>va</sub>l<sub>ue, an</sub>d i<sub>n</sub>iti<sub>a</sub>li<sub>zes</sub> th<sub>e va</sub>l<sub>ue</sub> l<sub>e</sub>d<sub>ger</sub> b<sub>y</sub> $\widehat { V } _ { i } ( s _ { 0 } ) : = v _ { \phi , i } ( q )$ <sub>.</sub> Th<sub>e</sub> <sub>sa</sub>m<sub>e</sub> r<sub>oo</sub>t <sub>va</sub>l<sub>ues</sub> initi<sub>a</sub>liz<sub>e</sub> th<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>o</sub>n <sub>pos</sub>t<sub>e</sub>ri<sub>o</sub>r<sub>.</sub> At <sub>eac</sub>h n<sub>o</sub>nt<sub>erm</sub>i<sub>na</sub>l <sub>pre</sub>fi<sub>x</sub> $s _ { t } ,$ <sub>,</sub> the service returns advertiser-conditioned lo<sub>g</sub>its. Th<sub>e p</sub>l<sub>a</sub>tf<sub>orm conver</sub>t<sub>s</sub> th<sub>em</sub> i<sub>n</sub>t<sub>o</sub> l<sub>oca</sub>l <sub>a</sub>d<sub>van</sub>t<sub>ages</sub> $\widehat { A } _ { \theta , i } ( s _ { t } , a ) \ =$ $\beta \log ( \pi _ { \theta , i } ( \boldsymbol { a } \mid \boldsymbol { s } _ { t } ) / \pi _ { \mathrm { r e f } } ( \boldsymbol { a } \mid \boldsymbol { s } _ { t } ) )$ <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> i<sub>n</sub>t<sub>o c</sub>hild<sub>-va</sub>l<sub>ue repor</sub>t<sub>s</sub> $\widehat { V } _ { i } ^ { s _ { t } } \big ( [ s _ { t } , a ] \big ) = \widehat { V } _ { i } ( s _ { t } ) + \widehat { A } _ { \theta , i } ( s _ { t } , a ) .$ <sup>7</sup> Th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>s</sub>t<sub>eps</sub> f<sub>o</sub>ll<sub>ow</sub> th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> di<sub>rec</sub>tl<sub>y.</sub> Si<sub>nce</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e a</sub>d<sub>ver</sub>ti<sub>ser se</sub>t i<sub>s</sub> t<sub>yp</sub>i<sub>ca</sub>ll<sub>y sma</sub>ll<sub>,</sub> LAMA i<sub>ncurs on</sub>l<sub>y a sma</sub>ll <sub>cons</sub>t<sub>an</sub>t<sub>-</sub>f<sub>ac</sub>t<sub>or over</sub>h<sub>ea</sub>d in lan<sub>g</sub>ua<sub>g</sub>e model forward <sub>p</sub>asses relative to <sub>g</sub>eneratin<sub>g</sub> a reference answer<sub>,</sub> kee<sub>p</sub>in<sub>g</sub> online servin<sub>g</sub> li<sub>g</sub>htwei<sub>g</sub>ht. Al<sub>g</sub>orithm 1 formall<sub>y</sub> summarizes the servin<sub>g p</sub>rocedure.

N<sub>o</sub>t<sub>a</sub>bl<sub>y,</sub> th<sub>e</sub> fr<sub>ac</sub>ti<sub>o</sub>n<sub>a</sub>l f<sub>o</sub>rm <sub>o</sub>f LAMA m<sub>ay c</sub>h<sub>a</sub>r<sub>ge a</sub>n <sub>a</sub>d<sub>ve</sub>rti<sub>se</sub>r that does not ultimatel<sub>y</sub> receive the advertisin<sub>g</sub> o<sub>pp</sub>ortunit<sub>y</sub>. We th<sub>ere</sub>f<sub>ore</sub> <sub>use</sub> th<sub>e</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t <sub>ran</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>se</sub>ttl<sub>emen</sub>t i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 1<sub>:</sub> <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> i<sub>s</sub> f<sub>orme</sub>d<sub>,</sub> th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>samp</sub>l<sub>es</sub> th<sub>e</sub> <sub>w</sub>i<sub>nner</sub> $I ^ { * } \sim \rho ( s _ { T } )$ <sub>,</sub> <sub>se</sub>ttl<sub>es</sub> <sub>on</sub>l<sub>y</sub> <sub>w</sub>ith th<sub>a</sub>t <sub>w</sub>i<sub>nner,</sub> <sub>an</sub>d <sub>c</sub>h<sub>arges</sub> <sub>zero</sub> t<sub>o</sub> <sub>a</sub>ll <sub>non-</sub> <sub>w</sub>inn<sub>e</sub>r<sub>s.</sub> B<sub>ecause</sub> thi<sub>s p</sub>r<sub>ese</sub>r<sub>ves e</sub>x<sub>pec</sub>t<sub>e</sub>d <sub>u</sub>tiliti<sub>es,</sub> th<sub>e</sub> in<sub>ce</sub>nti<sub>ve</sub>- com<sub>p</sub>atibilit<sub>y</sub> <sub>g</sub>uarantees carr<sub>y</sub> over directl<sub>y</sub>. It also <sub>y</sub>ields outcome IR <sub>w</sub>hil<sub>e preserv</sub>i<sub>ng expec</sub>t<sub>e</sub>d <sub>wea</sub>k b<sub>u</sub>d<sub>ge</sub>t b<sub>a</sub>l<sub>ance, even</sub> th<sub>oug</sub>h individual <sub>p</sub>a<sub>y</sub>ment transfers ma<sub>y</sub> have either si<sub>g</sub>n.

Proposition 1 (Outcome IR and Expected WBB). Ifthe served reports are exact, then the winner-settled LAMA implementation in Algorithm 1 is outcome individually rational: for every realized terminal state and sampled winner, non-winners pay zero and the winning advertiser � obtains non-negative realized utility. Moreover, conditional on the query and candidate advertiser set, the platform is weakly budget balanced in expectation: $\begin{array} { r } { \mathbb { E } [ \sum _ { i \in N } P _ { i } ( s _ { T } , I ^ { * } ) ] \ge 0 } \end{array}$

Algorithm 1 Latent Advertiser Mixture Auction (LAMA)   
Require: Query $q ,$ C<sub>an</sub>did<sub>a</sub>t<sub>e a</sub>d<sub>ver</sub>ti<sub>sers</sub> $N = \{ 1 , \ldots , n \}$ <sub>,</sub> R<sub>e</sub>f<sub>er-</sub>   
ence <sub>p</sub>o<sup>li</sup>c<sub>y</sub> $\pi _ { \mathrm { r e f } } ,$ Sh<sub>are</sub>d <sub>repor</sub>t <sub>mo</sub>d<sub>e</sub>l $( \pi _ { \theta } , v _ { \phi } ) _ { : }$ , <sup>T</sup>em<sub>p</sub>erature $\beta$   
1: Initialize: Context $s _ { 0 }  q ,$ <sup>S</sup>te<sub>p</sub> $t \gets 0 ,$ All<sub>oca</sub>ti<sub>o</sub>n <sub>pos</sub>t<sub>e</sub>ri<sub>o</sub>r   
$\begin{array} { r } { \rho _ { i } ( s _ { 0 } ) \gets \frac { \exp ( v _ { \phi , i } ( q ) / \beta ) } { \sum _ { j \in N } \exp ( v _ { \phi , j } ( q ) / \beta ) } } \end{array}$ f<sub>or a</sub>ll $i \in N$   
2: Initialize Ledger: $\hat { V } _ { i } ( s _ { 0 } ) \gets v _ { \phi , i } ( q )$ f<sub>or a</sub>ll $i \in N$   
3: while $s _ { t }$ is not terminal do   
4: for each advertiser $i \in N$ do   
5: // Compute local advantage and consistent child  
value reports   
6: $\begin{array} { r } { \hat { A } _ { \theta , i } ( s _ { t } , a ) \gets \beta \log \frac { \pi _ { \theta , i } ( a \vert s _ { t } ) } { \pi _ { \mathrm { r e f } } ( a \vert s _ { t } ) } } \end{array}$ ∀� $\in \mathcal { V }$   
7: $\hat { V } _ { i } ^ { s t } ( [ s _ { t } , a ] ) \gets \hat { V } _ { i } ( s _ { t } ) + \hat { A } _ { \theta , i } ( s _ { t } , a )$ ∀� $\in \mathcal { V }$   
8: end for   
9: // Hierarchical Sampling Process   
10: S<sub>amp</sub>l<sub>e</sub> <sub>a</sub> l<sub>a</sub>t<sub>en</sub>t <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> $I \sim \rho ( s _ { t } ) \qquad \circ$ Choose <sub>g</sub>uidin<sub>g</sub>   
<sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub>   
11: S<sub>amp</sub>l<sub>e nex</sub>t t<sub>o</sub>k<sub>en</sub> $a _ { t } \sim \hat { \pi } _ { I } ( \cdot \mid s _ { t } )$ <sub>⊲</sub> G<sub>enera</sub>t<sub>e</sub> t<sub>o</sub>k<sub>en</sub> f<sub>rom</sub>   
th<sub>a</sub>t <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub>’<sub>s po</sub>li<sub>cy</sub>   
12: // Update state and Bayesian allocation posterior   
13: $s _ { t + 1 } \gets [ s _ { t } , a _ { t } ]$   
14: for each advertiser $i \in N$ do   
15: $\begin{array} { r } { \rho _ { i } ( s _ { t + 1 } ) \gets \frac { \rho _ { i } ( s _ { t } ) \cdot \pi _ { \theta , i } ( a _ { t } | s _ { t } ) } { x ( a _ { t } | s _ { t } , \hat { V } ^ { s _ { t } } , \rho ( s _ { t } ) ) } } \end{array}$ <sub>⊲</sub> P<sub>os</sub>t<sub>e</sub>ri<sub>o</sub>r <sub>up</sub>d<sub>a</sub>t<sub>e</sub>   
16: $\hat { V } _ { i } ( s _ { t + 1 } ) \gets \hat { V } _ { i } ( s _ { t } ) + \hat { A } _ { \theta , i } ( s _ { t } , a _ { t } )$ <sub>⊲</sub> U<sub>p</sub>d<sub>a</sub>t<sub>e va</sub>l<sub>ue</sub> l<sub>e</sub>d<sub>ger</sub>   
17: end for   
18: $t \gets t + 1$   
19: end while   
20: Terminal State Reached: Let $s _ { T }$ b<sub>e</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> T<sub>o</sub>t<sub>a</sub>l   
t<sub>o</sub>k<sub>ens genera</sub>t<sub>e</sub>d �   
21<sub>:</sub> // Winner-Pay Settlement: Sample a single winner from   
the final posterior   
22<sub>:</sub> Sam<sub>p</sub>le a <sub>w</sub>inner $I ^ { * } \sim \rho ( s _ { T } )$   
23: for each advertiser $i \in N$ do   
24: $\mathbf { i } \mathbf { f } \ i = I ^ { * }$ then   
25: // Telescoped tra<sub>j</sub>ectory-level payment   
26: $\begin{array} { r l r } { P _ { i } } & { { }  } & { \rho _ { i } ( s _ { T } ) \hat { V } _ { i } ( s _ { T } ) - \beta \log \sum _ { j \in N } \exp ( \hat { V } _ { j } ( s _ { 0 } ) / \beta ) + } \end{array}$   
� log $\begin{array} { r } { \Big ( 1 + \sum _ { j \neq i } \exp ( \widehat { V } _ { j } ( s _ { 0 } ) / \beta ) \Big ) } \end{array}$   
27: $P _ { i } \gets \rho _ { i } ( s _ { T } ) ^ { - 1 } P _ { i } \qquad \mathsf { ~ \Rightarrow ~ A c }$ <sup>d</sup>just <sup>b</sup>y winning pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity   
28: else   
29: $P _ { i } \gets 0$ ⊲ Non-winners <sub>p</sub>a<sub>y</sub> nothin<sub>g</sub>   
30: end if   
31: end for   
32: return Generated sequence $s _ { T } ,$ Winner �<sup>∗</sup><sub>,</sub> Pa<sub>y</sub>ments   
$( P _ { 1 } , \ldots , P _ { n } )$

## 5 Experiments

In this section<sub>,</sub> <sub>w</sub>e em<sub>p</sub>iricall<sub>y</sub> e<sub>v</sub>al<sub>u</sub>ate the <sub>p</sub>ro<sub>p</sub>osed LAMA in <sub>g</sub>eneration-native advertisin<sub>g</sub> settin<sub>g</sub>s. The <sub>g</sub>oal is to test whether token-level <sub>p</sub>artici<sub>p</sub>ation of advertisers can im<sub>p</sub>rove monetization while <sub>p</sub>reservin<sub>g</sub> the <sub>q</sub>ualit<sub>y</sub> and naturalness of <sub>g</sub>enerated res<sub>p</sub>onses.

## 5.1 Setup

We e<sub>v</sub>al<sub>u</sub>ate mechanisms across re<sub>p</sub>resentati<sub>v</sub>e ad<sub>v</sub>ertisin<sub>g</sub> <sub>v</sub>erticals. E<sub>ac</sub>h <sub>ver</sub>ti<sub>ca</sub>l <sub>cons</sub>i<sub>s</sub>t<sub>s o</sub>f <sub>a se</sub>t <sub>o</sub>f <sub>user quer</sub>i<sub>es, can</sub>did<sub>a</sub>t<sub>e a</sub>d<sub>ver</sub>ti<sub>sers</sub> <sub>an</sub>d <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>ons.</sub> W<sub>e</sub> <sub>use</sub> <sub>a</sub> <sub>re</sub>f<sub>erence</sub> l<sub>anguage</sub> <sub>mo</sub>d<sub>e</sub>l (Qwen3-14B [29]), advertiser-shared re<sub>p</sub>ort models (trained LoRA heads [18]), and a <sub>p</sub>rediction model for oracle im<sub>p</sub>ression value [21, 31]. The main ex<sub>p</sub>eriments cover three verticals constructed from <sub>rea</sub>l<sub>-wor</sub>ld <sub>commerc</sub>i<sub>a</sub>l <sub>searc</sub>h <sub>query sp</sub>lit<sub>s</sub> i<sub>n</sub> th<sub>e</sub> W<sub>e</sub>bi<sub>s</sub> G<sub>enera</sub>t<sub>e</sub>d Native Ads 2024 dataset [24]: Workout, Vacation, and Car. Each <sub>ver</sub>ti<sub>ca</sub>l <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>ree</sub> h<sub>e</sub>t<sub>erogeneous</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>sers.</sub>

Baselines. We first compare against six heuristic incentive com-<sub>pa</sub>tibl<sub>e</sub> b<sub>ase</sub>li<sub>nes o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y com</sub>bi<sub>n</sub>i<sub>ng</sub> t<sub>wo auc</sub>ti<sub>on ru</sub>l<sub>es w</sub>ith three generation rules. The auction rules are: (i) Before. The winner i<sub>s c</sub>h<sub>osen</sub> b<sub>e</sub>f<sub>ore genera</sub>ti<sub>on un</sub>if<sub>orm</sub>l<sub>y a</sub>t <sub>ran</sub>d<sub>om, an</sub>d th<sub>e a</sub>d<sub>ver</sub>ti<sub>ser</sub> pays nothing. (ii) After. The candidate responses are generated first. Then advertisers are scored usin<sub>g p</sub>redicted im<sub>p</sub>ression value<sub>,</sub> and a second-price auction is applied. The generation rules are: (i) Original. Generate only with the reference model. (ii) Edit. Use a <sub>no-a</sub>d <sub>re</sub>f<sub>erence</sub> <sub>response</sub> <sub>as</sub> th<sub>e</sub> b<sub>ase</sub> <sub>an</sub>d i<sub>nser</sub>t <sub>sponsore</sub>d <sub>con</sub>t<sub>en</sub>t of an advertiser through a fused editing process. (iii) Policy. Each advertiser submits its <sub>p</sub>referred res<sub>p</sub>onse<sub>,</sub> i.e.<sub>,</sub> the content it ex<sub>p</sub>ects t<sub>o</sub> m<sub>a</sub>ximiz<sub>e</sub> it<sub>s</sub> im<sub>p</sub>r<sub>ess</sub>i<sub>o</sub>n <sub>va</sub>l<sub>ue.</sub> W<sub>e</sub> <sub>app</sub>r<sub>o</sub>xim<sub>a</sub>t<sub>e</sub> thi<sub>s</sub> r<sub>espo</sub>n<sub>se</sub> usin<sub>g</sub> best-of-K sam<sub>p</sub>lin<sub>g</sub> from the advertiser’s o<sub>p</sub>timal <sub>p</sub>olic<sub>y,</sub> with <sub>can</sub>did<sub>a</sub>t<sub>es</sub> <sub>ran</sub>k<sub>e</sub>d b<sub>y</sub> <sub>se</sub>lf<sub>-pre</sub>di<sub>c</sub>t<sub>e</sub>d i<sub>mpress</sub>i<sub>on</sub> <sub>va</sub>l<sub>ue.</sub>

We also com<sub>p</sub>are a<sub>g</sub>ainst MOSAIC [25], a re<sub>p</sub>resentative res<sub>p</sub>onselevel a<sub>gg</sub>re<sub>g</sub>ation mechanism. It <sub>g</sub>enerates � com<sub>p</sub>lete contextaware candidate res<sub>p</sub>onses<sub>,</sub> scores each candidate b<sub>y</sub> a<sub>gg</sub>re<sub>g</sub>ate <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> <sub>va</sub>l<sub>ue,</sub> <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>a</sub> fi<sub>na</sub>l <sub>response</sub> <sub>v</sub>i<sub>a</sub> <sub>a</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d l<sub>o</sub>tt<sub>ery.</sub>

Metrics. Our evaluation considers three perspectives: the platf<sub>orm,</sub> th<sub>e</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser,</sub> <sub>an</sub>d th<sub>e</sub> <sub>user.</sub> O<sub>n</sub> th<sub>e</sub> <sub>p</sub>l<sub>a</sub>tf<sub>orm</sub> <sub>s</sub>id<sub>e,</sub> <sub>we</sub> <sub>repor</sub>t (i) Welfare, defined as the realized advertising value minus the KL re<sub>g</sub>ularization term that ca<sub>p</sub>tures the user-ex<sub>p</sub>erience cost<sub>,</sub> and (ii) Revenue, defined as the payment collected from the winning advertiser. On the advertiser side, we measure (i) Value, namely the im<sub>p</sub>ression value obtained b<sub>y</sub> the winnin<sub>g</sub> advertiser. On the user side, we evaluate (i) Quality. Since the KL penalty alone d<sub>oes no</sub>t f<sub>u</sub>ll<sub>y re</sub>fl<sub>ec</sub>t th<sub>e ac</sub>t<sub>ua</sub>l <sub>user exper</sub>i<sub>ence o</sub>f th<sub>e genera</sub>t<sub>e</sub>d text, we ado<sub>p</sub>t the evaluation rubrics from the GEM-Bench [19] to assess user utility. Specifically, Quality aggregates multiple dimen-<sub>s</sub>i<sub>ons,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng answer re</sub>l<sub>evance,</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>veness,</sub> fl<sub>uency an</sub>d <sub>co</sub>h<sub>erence, a</sub>d <sub>na</sub>t<sub>ura</sub>l<sub>ness, a</sub>d f<sub>ac</sub>t<sub>ua</sub>lit<sub>y, an</sub>d i<sub>n</sub>t<sub>rus</sub>i<sub>veness.</sub>

## 5.2 Main Results

T<sub>a</sub>bl<sub>e</sub> 1 <sub>summar</sub>i<sub>zes</sub> th<sub>e ma</sub>i<sub>n resu</sub>lt<sub>s.</sub> LAMA <sub>ac</sub>hi<sub>eves</sub> th<sub>e s</sub>t<sub>ronges</sub>t <sub>overa</sub>ll <sub>per</sub>f<sub>ormance among</sub> th<sub>e compare</sub>d <sub>me</sub>th<sub>o</sub>d<sub>s, a</sub>tt<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> hi<sub>g</sub>hest mean <sub>p</sub>latform welfare (0.5205), revenue (0.8305), advertiser value (0.8568), and user <sub>q</sub>ualit<sub>y</sub> (66.5239). Relative to the allocateafter <sub>p</sub>olic<sub>y</sub> baseline<sub>,</sub> LAMA delivers a clear im<sub>p</sub>rovement in rev-<sub>enue,</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith f<sub>ur</sub>th<sub>er</sub> <sub>ga</sub>i<sub>ns</sub> i<sub>n</sub> <sub>we</sub>lf<sub>are</sub> <sub>an</sub>d <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> <sub>va</sub>l<sub>ue.</sub> At the same time<sub>,</sub> it <sub>p</sub>reserves user-facin<sub>g</sub> res<sub>p</sub>onse <sub>q</sub>ualit<sub>y,</sub> matchin<sub>g</sub> the best-<sub>p</sub>erformin<sub>g</sub> baseline on this dimension. A<sub>pp</sub>endix D <sub>p</sub>ro-<sub>v</sub>id<sub>es a</sub>dditi<sub>ona</sub>l <sub>exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>ev</sub>id<sub>ence,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng case s</sub>t<sub>u</sub>di<sub>es an</sub>d t<sub>o</sub>k<sub>en-</sub>l<sub>eve</sub>l <sub>ana</sub>l<sub>yses</sub> f<sub>or</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>ary</sub> <sub>an</sub>d <sub>compe</sub>ti<sub>ng</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>sers,</sub> latenc<sub>y</sub> measurements<sub>,</sub> value <sub>p</sub>rediction accurac<sub>y</sub> and calibration<sub>,</sub> <sub>an</sub>d bid<sub>-o</sub>f<sub>se</sub>t <sub>sweeps</sub> <sub>o</sub>f <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> <sub>paymen</sub>t<sub>,</sub> <sub>an</sub>d <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> <sub>u</sub>tilit<sub>y.</sub>

Table 1: Main results on three representative verticals from Webis. Smaller gray values denote 95% bootstrap errors.
<table><tr><td rowspan="2">Method</td><td colspan="2">PLATFORM</td><td>USER</td><td>ADVERTISER</td></tr><tr><td>Wel. ↑</td><td>Rev. ↑</td><td>Qua. ↑</td><td>Val. ↑</td></tr><tr><td></td><td>Allocate before generation</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Reference</td><td>0.2945</td><td>0.0000</td><td>52.9429</td><td>0.2945</td></tr><tr><td>± 0.0117</td><td>± 0.0000</td><td>± 0.2753</td><td>± 0.0116</td></tr><tr><td>Edit</td><td>0.0297</td><td>0.0000</td><td>65.7685</td><td>0.2598</td></tr><tr><td rowspan="2">Policy</td><td>± 0.0138</td><td>± 0.0000</td><td>± 0.2323</td><td>± 0.0133</td></tr><tr><td>0.4562</td><td>0.0000</td><td>65.1635</td><td>0.8225</td></tr><tr><td></td><td>± 0.0070</td><td>± 0.0000</td><td>± 0.2230</td><td>± 0.0045</td></tr><tr><td></td><td>Allocate after generation</td><td></td><td></td><td></td></tr><tr><td>Reference</td><td>0.3558</td><td>0.2875</td><td>53.8997</td><td>0.3558</td></tr><tr><td rowspan="2">Edit</td><td>± 0.0134</td><td>± 0.0112</td><td>±0.2642</td><td>± 0.0132</td></tr><tr><td>0.1772</td><td>0.2490</td><td>66.4785</td><td>0.4046</td></tr><tr><td></td><td>± 0.0135</td><td>± 0.0117</td><td>±0.2423</td><td>± 0.0132</td></tr><tr><td rowspan="2">Policy</td><td>0.5080</td><td>0.7501</td><td>65.5939</td><td>0.8253</td></tr><tr><td>± 0.0041</td><td>± 0.0053</td><td>± 0.2197</td><td>± 0.0044</td></tr><tr><td>MOSAIC</td><td>0.4390 ± 0.0118</td><td>0.5274 ± 0.0121</td><td>60.4100 ± 0.3236</td><td>0.5550 ± 0.0118</td></tr><tr><td rowspan="2"></td><td></td><td></td><td></td><td></td></tr><tr><td>0.5205</td><td>0.8305</td><td>66.5239</td><td>0.8568</td></tr><tr><td>LAMA (Ours)</td><td>±0.0053</td><td>±0.0049</td><td>±0.2278</td><td>±0.0043</td></tr></table>

Taken to<sub>g</sub>ether<sub>,</sub> these results <sub>p</sub>rovide an initial <sub>p</sub>roof of conce<sub>p</sub>t f<sub>o</sub>r LAMA: t<sub>o</sub>k<sub>e</sub>n-l<sub>eve</sub>l <sub>a</sub>d<sub>ve</sub>rti<sub>se</sub>r <sub>pa</sub>rti<sub>c</sub>i<sub>pa</sub>ti<sub>o</sub>n <sub>ca</sub>n im<sub>p</sub>r<sub>ove</sub> m<sub>o</sub>n<sub>e</sub>tization without sacrificin<sub>g</sub> res<sub>p</sub>onse <sub>q</sub>ualit<sub>y,</sub> su<sub>gg</sub>estin<sub>g</sub> the <sub>p</sub>otential of <sub>g</sub>eneration-native advertisin<sub>g</sub> as a viable <sub>p</sub>aradi<sub>g</sub>m.

## 6 Related Work

Classical online advertisin<sub>g</sub> auctions t<sub>yp</sub>icall<sub>y</sub> allocate exo<sub>g</sub>enousl<sub>y</sub> d<sub>e</sub>fi<sub>ne</sub>d i<sub>nven</sub>t<sub>ory,</sub> <sub>suc</sub>h <sub>as</sub> <sub>sponsore</sub>d<sub>-searc</sub>h <sub>or</sub> di<sub>sp</sub>l<sub>ay</sub> <sub>s</sub>l<sub>o</sub>t<sub>s,</sub> th<sub>roug</sub>h mechanisms includin<sub>g</sub> GSP and VCG [10, 27]. Subse<sub>q</sub>uent work in-<sub>corpora</sub>t<sub>es casca</sub>d<sub>e e</sub>f<sub>ec</sub>t<sub>s, a</sub>d <sub>ex</sub>t<sub>erna</sub>liti<sub>es, an</sub>d <sub>r</sub>i<sub>c</sub>h<sub>er c</sub>li<sub>c</sub>k<sub>-</sub>th<sub>roug</sub>h rate structures [2, 7, 11], and studies the joint o<sub>p</sub>timization of advertisin<sub>g</sub> and or<sub>g</sub>anic content across search<sub>,</sub> recommendation<sub>,</sub> ecommerce, and contextual advertisin<sub>g</sub> [3, 6, 12–14, 20]. Des<sub>p</sub>ite <sub>r</sub>i<sub>c</sub>h<sub>er</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>ons w</sub>ith <sub>users an</sub>d <sub>surroun</sub>di<sub>ng con</sub>t<sub>en</sub>t<sub>,</sub> th<sub>ese mo</sub>d<sub>-</sub> <sub>e</sub>l<sub>s</sub> l<sub>arge</sub>l<sub>y re</sub>t<sub>a</sub>i<sub>n a common a</sub>b<sub>s</sub>t<sub>rac</sub>ti<sub>on:</sub> th<sub>e mec</sub>h<sub>an</sub>i<sub>sm a</sub>ll<sub>oca</sub>t<sub>es</sub> <sub>p</sub>re-existin<sub>g</sub> slots<sub>,</sub> items<sub>,</sub> or <sub>p</sub>ositions whose advertisin<sub>g</sub> o<sub>pp</sub>ortuniti<sub>es are</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>e</sub>f<sub>ore a</sub>ll<sub>oca</sub>ti<sub>on.</sub>

G<sub>enera</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>er</sub>f<sub>aces c</sub>h<sub>a</sub>ll<sub>enge</sub> thi<sub>s a</sub>b<sub>s</sub>t<sub>rac</sub>ti<sub>on</sub> b<sub>ecause a</sub>d<sub>ver-</sub> tisin<sub>g</sub> o<sub>pp</sub>ortunities ma<sub>y</sub> de<sub>p</sub>end on the res<sub>p</sub>onse bein<sub>g g</sub>enerated. M<sub>ore</sub> b<sub>roa</sub>dl<sub>y, recen</sub>t <sub>wor</sub>k <sub>s</sub>t<sub>u</sub>di<sub>es mec</sub>h<sub>an</sub>i<sub>sms</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>c</sub> a<sub>g</sub>ents influence LLM out<sub>p</sub>uts throu<sub>g</sub>h next-token distributions or <sub>p</sub>references<sub>,</sub> makin<sub>g</sub> <sub>g</sub>eneration itself <sub>p</sub>art of the mechanism outcome [8, 9, 25]. In LLM advertisin<sub>g</sub>, some works define o<sub>pp</sub>ortunities over discourse se<sub>g</sub>ments or context-de<sub>p</sub>endent <sub>p</sub>osition– creative <sub>p</sub>airs [4, 15], while others stud<sub>y</sub> res<sub>p</sub>onse-level ad insertion [28]. More LLM-native a<sub>pp</sub>roaches incor<sub>p</sub>orate res<sub>p</sub>onse-<sub>q</sub>ualit<sub>y</sub> constraints [16], allocate advertisin<sub>g</sub> throu<sub>g</sub>h the model’s out<sub>p</sub>ut distribution [32], or ex<sub>p</sub>lore o<sub>pp</sub>ortunities arisin<sub>g</sub> from retrieval decisions and internal model re<sub>p</sub>resentations [26, 30]. To<sub>g</sub>ether, th<sub>ese wor</sub>k<sub>s progress</sub>i<sub>ve</sub>l<sub>y move a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng a</sub>ll<sub>oca</sub>ti<sub>on</sub> f<sub>rom</sub> fi<sub>xe</sub>d <sub>s</sub>l<sub>o</sub>t<sub>s</sub> t<sub>owar</sub>d th<sub>e genera</sub>ti<sub>on process</sub> it<sub>se</sub>lf<sub>.</sub>

O<sub>ur wor</sub>k t<sub>a</sub>k<sub>es a</sub> f<sub>ur</sub>th<sub>er s</sub>t<sub>ep</sub> t<sub>owar</sub>d <sub>genera</sub>ti<sub>on-na</sub>ti<sub>ve a</sub>d<sub>ver-</sub> tisin<sub>g</sub> b<sub>y</sub> allowin<sub>g</sub> advertisers to influence <sub>g</sub>eneration at tokenlevel <sub>g</sub>ranularit<sub>y</sub>. Rather than allocatin<sub>g</sub> an o<sub>pp</sub>ortunit<sub>y</sub> identified before or after <sub>g</sub>eneration<sub>,</sub> LAMA o<sub>p</sub>erates alon<sub>g</sub> the <sub>g</sub>eneration trajectory: a<sup>d</sup>vertiser in<sup>fl</sup>uence evo<sup>l</sup>ves wit<sup>h</sup> t<sup>h</sup>e current pre<sup>fi</sup>x, <sub>p</sub>osterior beliefs<sub>,</sub> and continuation values. Thus<sub>,</sub> the advertisin<sub>g</sub> o<sub>p</sub>- <sub>p</sub>ortunit<sub>y</sub> itself emer<sub>g</sub>es d<sub>y</sub>namicall<sub>y</sub> throu<sub>g</sub>h <sub>g</sub>eneration<sub>,</sub> shiftin<sub>g</sub> th<sub>e</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm-</sub>d<sub>es</sub>i<sub>gn</sub> <sub>pro</sub>bl<sub>em</sub> f<sub>rom</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>ng</sub> <sub>sponsore</sub>d <sub>con</sub>t<sub>en</sub>t within <sub>g</sub>enerated res<sub>p</sub>onses to <sub>g</sub>overnin<sub>g</sub> how advertiser influence <sup>sha</sup>p<sup>es</sup> <sup>the</sup> <sup>res</sup>p<sup>onse-</sup>g<sup>eneration</sup> p<sup>rocess.</sup>

## 7 Conclusion

In this work<sub>,</sub> we introduced <sub>g</sub>eneration-native advertisin<sub>g,</sub> where advertisin<sub>g</sub> o<sub>pp</sub>ortunities are sha<sub>p</sub>ed durin<sub>g</sub> <sub>g</sub>eneration rather than <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub>d <sub>on</sub>l<sub>y a</sub>ft<sub>er</sub> th<sub>ey</sub> h<sub>ave</sub> b<sub>een</sub> f<sub>orme</sub>d<sub>.</sub> LAMA i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>es</sub> thi<sub>s</sub> idea at token level<sub>,</sub> achievin<sub>g</sub> Markov DSIC and IR with bounded <sub>we</sub>lf<sub>are</sub> l<sub>oss,</sub> t<sub>oge</sub>th<sub>er w</sub>ith <sub>a prac</sub>ti<sub>ca</sub>l l<sub>earn</sub>i<sub>ng-</sub>b<sub>ase</sub>d i<sub>mp</sub>l<sub>emen</sub>t<sub>a-</sub> ti<sub>o</sub>n<sub>.</sub> Pr<sub>oo</sub>f-<sub>o</sub>f-<sub>co</sub>n<sub>cep</sub>t <sub>e</sub>x<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t LAMA <sub>ca</sub>n im<sub>p</sub>r<sub>ove</sub> monetization <sub>w</sub>hile maintainin<sub>g u</sub>ser-facin<sub>g</sub> res<sub>p</sub>onse <sub>qu</sub>alit<sub>y</sub>. More b<sub>roa</sub>dl<sub>y, our resu</sub>lt<sub>s po</sub>i<sub>n</sub>t t<sub>owar</sub>d <sub>a new</sub> di<sub>rec</sub>ti<sub>on</sub> f<sub>or a</sub>d<sub>ver</sub>ti<sub>s</sub>i<sub>ng</sub> mechanism desi<sub>g</sub>n in <sub>g</sub>enerative interfaces<sub>,</sub> where mechanisms <sub>g</sub>overn not onl<sub>y</sub> who receives an advertisin<sub>g</sub> o<sub>pp</sub>ortunit<sub>y,</sub> but also how <sup>that o</sup>pp<sup>ortunit</sup>y <sup>emer</sup>g<sup>es throu</sup>g<sup>h</sup> g<sup>eneration.</sup>

## References

[1] Ga<sub>g</sub>an A<sub>gg</sub>arwal, Ashwinkumar Badanidi<sub>y</sub>uru, and Aran<sub>y</sub>ak Mehta. 2019. Autobidding with constraints. In International Conference on Web and Internet Economics. Springer, 17–30.

[2] Ga<sub>g</sub>an A<sub>gg</sub>arwal, Jon Feldman, S. Muthukrishnan, and Martin Pál. 2008. S<sub>p</sub>onsored Search Auctions with Markovian Users. In Internet and Network Economics, 4th International Workshop, WINE 2008, Shanghai, China, December 17-20, 2008. Proceedings (Lecture Notes in Computer Science), Christos H. Papadimitriou and Shuzhon<sub>g</sub> Zhan<sub>g</sub> (Eds.). S<sub>p</sub>rin<sub>g</sub>er, 621–628. doi:10.1007/978-3-540-92185-1\_68

[3] Nan An, Weian Li, Qi Qi, and Liang Zhang. 2025. Merging Mechanisms for Ads and Organic Items in E-commerce Platforms. Proceedings ofthe AAAI Conference on Artificial Intelligence 39, 13 (April 2025), 13547–13554. doi:10.1609/aaai.v39i13. 33479

[4] Santia<sub>g</sub>o Balseiro, Kshi<sub>p</sub>ra Bhawalkar, Yuan Den<sub>g</sub>, Zhe Fen<sub>g</sub>, Jiemin<sub>g</sub> Mao, Ar<sub>a</sub>n<sub>ya</sub>k M<sub>e</sub>ht<sub>a,</sub> V<sub>a</sub>h<sub>a</sub>b Mirr<sub>o</sub>kni<sub>,</sub> R<sub>e</sub>n<sub>a</sub>t<sub>o</sub> P<sub>aes</sub> L<sub>e</sub>m<sub>e,</sub> Di W<sub>a</sub>n<sub>g, a</sub>nd S<sub>o</sub>n<sub>g</sub> Z<sub>uo.</sub> 2025. Position Auctions in AI-Generated Content. arXiv:2506.03309 [cs.GT] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2506.03309

[5] Ral<sub>p</sub>h Allan Bradle<sub>y</sub> and Milton E Terr<sub>y</sub>. 1952. Rank anal<sub>y</sub>sis of incom<sub>p</sub>lete block designs: I. the method of paired comparisons. Biometrika 39, 3/4 (1952), 324–345.

[6] Carlos Carrion, Zenan Wan<sub>g</sub>, Harikesh Nair, Xian<sub>g</sub>hon<sub>g</sub> Luo, Yulin Lei, Pei<sub>q</sub>in Gu, Xilian<sub>g</sub> Lin, Wenlon<sub>g</sub> Chen, Junshen<sub>g</sub> Jin, Fanan Zhu, Chan<sub>gp</sub>in<sub>g</sub> Pen<sub>g</sub>, Yongjun Bao, Zhangang Lin, Weipeng Yan, and Jingping Shao. 2023. Blending Advertising with Organic Content in E-commerce via Virtual Bids. Proceedings ofthe AAAI Conference on Artificial Intelligence 37, 13 (June 2023), 15476–15484. d<sub>o</sub>i:10<sub>.</sub>1609/<sub>aaa</sub>i<sub>.v</sub>37i13<sub>.</sub>26835

[7] Ru<sub>gg</sub>iero Cavallo and Christo<sub>p</sub>her A. Wilkens. 2014. GSP with General Independent Click-through-Rates. In Web and Internet Economics - 10th International Conference, WINE 2014, Beijing, China, December 14-17, 2014. Proceedings (Lecture Notes in Computer Science), Tie-Yan Liu, Qi Qi, and Yinyu Ye (Eds.). Springer, 400–416. doi:10.1007/978-3-319-13129-0<sub>\_</sub>32

[8] Avinava Dube<sub>y</sub>, Zhe Fen<sub>g</sub>, Rahul Kidambi, Aran<sub>y</sub>ak Mehta, and Di Wan<sub>g</sub>. 2024. Auctions with LLM Summaries. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2024, Barcelona, Spain, August 25-29, 2024, Ricardo Baeza-Yates and Francesco Bonchi (Eds.). ACM, 713–722. doi:10.1145/3637528.3672022

[9] Paul Düttin<sub>g</sub>, Vahab Mirrokni, Renato Paes Leme, Haifen<sub>g</sub> Xu, and Son<sub>g</sub> Zuo. 2024. Mechanism Design for Large Language Models. In Proceedings of the ACM on Web Conference 2024, WWW 2024, Singapore, May 13-17, 2024, Tat-Seng Chua, Chon<sub>g</sub>-Wah N<sub>g</sub>o, Ravi Kumar, Had<sub>y</sub> W. Lauw, and Ro<sub>y</sub> Ka-Wei Lee (Eds.). ACM, 144–155. doi:10.1145/3589334.3645511

[10] Benjamin Edelman, Michael Ostrovsk<sub>y</sub>, and Michael Schwarz. 2007. Internet Ad<sub>ve</sub>rti<sub>s</sub>in<sub>g a</sub>nd th<sub>e</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>liz<sub>e</sub>d S<sub>eco</sub>nd-Pri<sub>ce</sub> A<sub>uc</sub>ti<sub>o</sub>n: S<sub>e</sub>llin<sub>g</sub> Billi<sub>o</sub>n<sub>s o</sub>f D<sub>o</sub>ll<sub>a</sub>r<sub>s</sub> Worth of Keywords. American Economic Review 97, 1 (Feb. 2007), 242–259. doi:10.1257/aer.97.1.242

[11] Nicola Gatti, Marco Rocco, Paolo Serafino, and Carmine Ventre. 2016. Towards Better Models of Externalities in Sponsored Search Auctions. In ECAI 2016 - 22nd European Conference on Artificial Intelligence, 29 August-2 September 2016, The Hague, The Netherlands - Including Prestigious Applications ofArtificial Intelligence (PAIS 2016) (Frontiers in Artificial Intelligence and Applications), Gal A. Kaminka, M<sub>a</sub>ri<sub>a</sub> F<sub>o</sub>x<sub>,</sub> P<sub>ao</sub>l<sub>o</sub> B<sub>ouque</sub>t<sub>,</sub> E<sub>y</sub>k<sub>e</sub> Hüll<sub>e</sub>rm<sub>e</sub>i<sub>e</sub>r<sub>,</sub> Vir<sub>g</sub>ini<sub>a</sub> Di<sub>g</sub>n<sub>u</sub>m<sub>,</sub> Fr<sub>a</sub>nk Di<sub>g</sub>n<sub>u</sub>m<sub>,</sub> and Frank van Harmelen (Eds.). IOS Press, 1167–1175. doi:10.3233/978-1-61499- 672-9-1167

[12] Nicola Gatti, Marco Rocco, Paolo Serafino, and Carmine Ventre. 2018. Towards better models of externalities in sponsored search auctions. Theor. Comput. Sci. 745 (2018), 150–162. doi:10.1016/J.TCS.2018.06.011

[13] Ar<sub>p</sub>ita Ghosh and Mohammad Mahdian. 2008. Externalities in online advertisin<sub>g</sub>. In Proceedings ofthe 17th International Conference on World Wide Web, WWW 2008, Beijing, China, April 21-25, 2008, Jinpeng Huai, Robin Chen, Hsiao-Wuen Hon, Yunhao Liu, Wei-Yin<sub>g</sub> Ma, Andrew Tomkins, and Xiaodon<sub>g</sub> Zhan<sub>g</sub> (Eds.). ACM<sub>,</sub> 161–168<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1145/1367497<sub>.</sub>1367520

[14] Ioannis Giotis and Anna R. Karlin. 2008. On the E<sub>q</sub>uilibria and Eficienc<sub>y</sub> of the GSP Mechanism in Keyword Auctions with Externalities. In Internet and Network Economics, 4th International Workshop, WINE 2008, Shanghai, China, December 17-20, 2008. Proceedings (Lecture Notes in Computer Science), Christos H. Pa<sub>p</sub>adimitriou and Shuzhon<sub>g</sub> Zhan<sub>g</sub> (Eds.). S<sub>p</sub>rin<sub>g</sub>er, 629–638. doi:10.1007/978- 3-540-92185-1<sub>\_</sub>69

[15] MohammadTa<sub>g</sub>hi Hajia<sub>g</sub>ha<sub>y</sub>i, Sébastien Lahaie, Keivan Rezaei, and Suho Shin. 2024. Ad Auctions for LLMs via Retrieval Au mented Generation. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Pa<sub>q</sub>uet, J. Tomczak, and C. Zhan<sub>g</sub> (Eds.), Vol. 37. Curran Associates, Inc.<sub>,</sub> 18445–18480. doi:10.52202/079017-0585

[16] Jiale Han and Xiaowu Dai. 2026. Mechanism Desi<sub>g</sub>n for Qualit<sub>y</sub>-Preservin<sub>g</sub> LLM Advertisin<sub>g</sub>. arXiv:2605.10964 [cs.GT] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2605.10964

[17] Yue He, Xiujun Chen, Di Wu, Junwei Pan, Qing Tan, Chuan Yu, Jian Xu, and Xiao<sub>q</sub>ian<sub>g</sub> Zhu. 2021. A unified solution to constrained biddin<sub>g</sub> in online dis<sub>p</sub>la<sub>y</sub> advertising. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2993–3001.

[18] Edward J Hu, Yelon<sub>g</sub> Shen, Philli<sub>p</sub> Wallis, Ze<sub>y</sub>uan Allen-Zhu, Yuanzhi Li, Shean Wan<sub>g,</sub> Lu Wan<sub>g,</sub> and Weizhu Chen. 2021. Lora: Low-rank ada<sub>p</sub>tation of lar<sub>g</sub>e language models. arXiv preprint arXiv:2106.09685 (2021).

[19] Silan Hu, Shi<sub>q</sub>i Zhan<sub>g</sub>, Yimin Shi, and Xiaokui Xiao. 2026. GEM-Bench: A Benc<sup>h</sup>mar<sup>k f</sup>or A<sup>d</sup>-Injecte<sup>d</sup> Response Generation wit<sup>h</sup>in Generative Engine Marketin<sub>g</sub>. arXiv:2509.14221 [cs.IR] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2509.14221

[20] Weian Li, Qi Qi, Changjun Wang, and Changyuan Yu. 2023. Optimally integrating ad auction into e-commerce platforms. Theoretical Computer Science 976 (Oct. 2023), 114141. doi:10.1016/j.tcs.2023.114141

[21] lv12. 2024. esci-ms-marco-MiniLM-L-12-v2. htt<sub>p</sub>s://hu<sub>gg</sub>in<sub>g</sub>face.co/lv12/escim<sub>s</sub>-m<sub>a</sub>r<sub>co</sub>-MiniLM-L-12-<sub>v</sub>2<sub>.</sub> Cr<sub>oss</sub>-<sub>e</sub>n<sub>co</sub>d<sub>e</sub>r fin<sub>e</sub>-t<sub>u</sub>n<sub>e</sub>d <sub>o</sub>n th<sub>e</sub> Am<sub>a</sub>z<sub>o</sub>n ESCI d<sub>a</sub>t<sub>ase</sub>t<sub>;</sub> b<sub>ase</sub>d <sub>o</sub>n <sub>c</sub>r<sub>oss</sub>-<sub>e</sub>n<sub>co</sub>d<sub>e</sub>r/m<sub>s</sub>-m<sub>a</sub>r<sub>co</sub>-MiniLM-L12-<sub>v</sub>2<sub>.</sub>

[22] Marshall McLuhan. 1964. Understanding media. Vol. 2002. Routledge London.

[23] Lon<sub>g</sub> Ou<sub>y</sub>an<sub>g</sub>, Jefre<sub>y</sub> Wu, Xu Jian<sub>g</sub>, Dio<sub>g</sub>o Almeida, Carroll Wainwri<sub>g</sub>ht, Pamela Mi<sub>s</sub>hkin<sub>,</sub> Ch<sub>o</sub>n<sub>g</sub> Zh<sub>a</sub>n<sub>g,</sub> S<sub>a</sub>ndhini A<sub>ga</sub>r<sub>wa</sub>l<sub>,</sub> K<sub>a</sub>t<sub>a</sub>rin<sub>a</sub> Sl<sub>a</sub>m<sub>a,</sub> Al<sub>e</sub>x R<sub>ay, e</sub>t <sub>a</sub>l<sub>.</sub> 2022<sub>.</sub> Training language models to follow instructions with human feedback. Advances in neural information processing systems 35 (2022), 27730–27744.

[24] Sebastian Schmidt, Ines Zelch, Janek Bevendorf, Benno Stein, Matthias Ha<sub>g</sub>en, <sub>a</sub>nd M<sub>a</sub>rtin P<sub>o</sub>tth<sub>as</sub>t<sub>.</sub> 2024<sub>.</sub> D<sub>e</sub>t<sub>ec</sub>tin<sub>g</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d N<sub>a</sub>ti<sub>ve</sub> Ad<sub>s</sub> in C<sub>o</sub>n<sub>ve</sub>r<sub>sa</sub>ti<sub>o</sub>n<sub>a</sub>l Search. In Companion Proceedings ofthe ACM Web Conference 2024. ACM, 722– 725. doi:10.1145/3589335.3651489

[25] Ermis Soumalias, Michael J. Curr<sub>y</sub>, and Sven Seuken. 2024. Truthful A<sub>gg</sub>re<sub>g</sub>ation of LLMs with an A lication to Online Advertisin . CoRR abs/2405.05905 (2024). <sub>a</sub>rXi<sub>v</sub>:2405<sub>.</sub>05905 d<sub>o</sub>i:10<sub>.</sub>48550/ARXIV<sub>.</sub>2405<sub>.</sub>05905

[26] Haoran Sun, Xinrui Son<sub>g</sub>, Xin<sub>y</sub>u Zhan<sub>g</sub>, Zhaohua Chen, Xu Chu, Zhilin Zhan<sub>g</sub>, Chuan Yu, Jian Xu, Bo Zhen<sub>g</sub>, and Xiaotie Den<sub>g</sub>. 2026. LERA: LLM-Enhanced RAG for Ad Auction in Generative Chatbots. arXiv:2605.16474 [cs.IR] htt s: //arxiv.or<sub>g</sub>/abs/2605.16474

[27] Hal R. Varian. 2007. Position Auctions. International Journal ofIndustrial Organization 25, 6 (Dec. 2007), 1163–1178. doi:10.1016/j.ijindorg.2006.10.002

[28] Shen wei Xu, Zhaohua Chen, Xiaotie Den , Zhi i Huan , and Grant S<sub>c</sub>h<sub>oene</sub>b<sub>ec</sub>k<sub>.</sub> 2026<sub>.</sub> Ad In<sub>se</sub>rti<sub>o</sub>n in LLM-G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>t<sub>e</sub>d R<sub>espo</sub>n<sub>ses.</sub> arXiv:2601.19435 [cs.GT] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2601.19435

[29] An Yan<sub>g</sub>, Anfen<sub>g</sub> Li, Baoson<sub>g</sub> Yan<sub>g</sub>, Beichen Zhan<sub>g</sub>, Bin<sub>y</sub>uan Hui, Bo Zhen<sub>g</sub>, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).

[30] Peiran Yun, Wenxin Xu, Jia<sub>y</sub>uan Liu, Yihan<sub>g</sub> Zhan<sub>g</sub>, Lian<sub>g</sub> Zen<sub>g</sub>, Lin<sub>g</sub>kai Kon<sub>g</sub>, <sub>a</sub>nd T<sub>o</sub>n<sub>g</sub>h<sub>a</sub>n W<sub>a</sub>n<sub>g.</sub> 2026<sub>.</sub> LLM Ad<sub>ve</sub>rti<sub>se</sub>m<sub>e</sub>nt b<sub>ase</sub>d <sub>o</sub>n N<sub>eu</sub>r<sub>o</sub>n A<sub>uc</sub>ti<sub>o</sub>n<sub>s.</sub> arXiv:2605.08326 [cs.LG] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2605.08326

[31] Yanzhao Zhan<sub>g</sub>, Min<sub>g</sub>xin Li, Din<sub>g</sub>kun Lon<sub>g</sub>, Xin Zhan<sub>g</sub>, Huan Lin, Baoson<sub>g</sub> Yan<sub>g</sub>, Pen<sub>g</sub>jun Xie, An Yan<sub>g</sub>, Da<sub>y</sub>ihen<sub>g</sub> Liu, Jun<sub>y</sub>an<sub>g</sub> Lin, Fei Huan<sub>g</sub>, and Jin<sub>g</sub>ren Zhou.

2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv preprint arXiv:2506.05176 (2025). doi:10.48550/arXiv. 2506.05176

[32] Chujie Zhao, Qun Hu, Shiping Song, Dagui Chen, Han Zhu, Jian Xu, and Bo Zh<sub>e</sub>n<sub>g.</sub> 2026<sub>.</sub> LLM-A<sub>uc</sub>ti<sub>o</sub>n: G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>ve</sub> A<sub>uc</sub>ti<sub>o</sub>n t<sub>owa</sub>rd<sub>s</sub> LLM-N<sub>a</sub>ti<sub>ve</sub> Ad<sub>ve</sub>rti<sub>s</sub>in<sub>g.</sub> arXiv:2512.10551 [cs.GT] htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2512.10551

## A Report-Model Training

Al<sub>g</sub>orithm 2 <sub>g</sub>ives a <sub>p</sub>ractical im<sub>p</sub>lementation of the decom<sub>p</sub>osed t<sub>w</sub>o learnin<sub>g</sub> tasks in Section 4.1.

## B Potential Business Format

Fi<sub>g</sub>ure 3 <sub>p</sub>resents three re<sub>p</sub>resentative business formats for a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> LAMA in the sin<sub>g</sub>le-winner settin<sub>g</sub>: a h<sub>yp</sub>erlink embedded in the <sub>genera</sub>t<sub>e</sub>d <sub>response,</sub> <sub>a</sub> <sub>c</sub>li<sub>c</sub>k<sub>a</sub>bl<sub>e</sub> <sub>response</sub> bl<sub>oc</sub>k<sub>,</sub> <sub>an</sub>d <sub>a</sub> <sub>sponsore</sub>d <sub>car</sub>d di<sub>sp</sub>l<sub>aye</sub>d <sub>a</sub>l<sub>ongs</sub>id<sub>e</sub> th<sub>e</sub> <sub>response.</sub>

## C Proofs

Lemma 1 (Optimal soft Bellman recursion). Consider a finitehorizon token MDP with deterministic transition $s _ { t + 1 } = \left[ s _ { t } , a _ { t } \right]$ , reference policy $\pi _ { \mathrm { r e f } }$ , and terminal reward $r : S _ { T } \to \mathbb { R }$ . For every state $s ,$ define the optimal KL-regularized continuation value

$$
V ^ { * } ( s ) : = \operatorname* { s u p } _ { \pi } \mathbb { E } _ { \tau \sim \pi ( \cdot \vert s ) } \left[ r ( s _ { T } ) - \beta \sum _ { t = 0 } ^ { T _ { \tau } - 1 } \log \frac { \pi ( a _ { t } \mid s _ { t } ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } \right] .
$$

Assume $\pi _ { \mathrm { r e f } } ( a \mid s ) > 0$ for every feasible action � at every nonterminal state �. Then $V ^ { * }$ satisfies the terminal boundary condition $V ^ { * } ( \ell ) = r ( \ell ) , \ell \in S _ { T } ,$ and the soft Bellman recursion

$$
V ^ { * } ( s ) = \beta \log \sum _ { a \in \mathcal { N } } \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( \frac { V ^ { * } ( [ s , a ] ) } { \beta } \right) , s \notin S _ { T } .
$$

Moreover, the optimal continuation policy is

$$
\pi ^ { * } ( a \mid s ) = \frac { \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( V ^ { * } ( [ s , a ] ) / \beta \right) } { \sum _ { a ^ { \prime } \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a ^ { \prime } \mid s ) \exp \left( V ^ { * } ( [ s , a ^ { \prime } ] ) / \beta \right) } .
$$

Proof<sub>.</sub> Th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>con</sub>diti<sub>on</sub> i<sub>s</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e.</sub> ${ \mathrm { I f ~ } } s \ = \ \ell \ \in \ S _ { T }$ th<sub>e process</sub> h<sub>as a</sub>l<sub>rea</sub>d<sub>y</sub> t<sub>erm</sub>i<sub>na</sub>t<sub>e</sub>d <sub>an</sub>d <sub>no</sub> f<sub>ur</sub>th<sub>er</sub> KL <sub>pena</sub>lt<sub>y</sub> i<sub>s</sub> i<sub>ncurre</sub>d<sub>, so</sub>

$$
V ^ { * } ( \ell ) = r ( \ell ) .
$$

Now fix a non-terminal state <sub>�</sub>. An<sub>y</sub> continuation <sub>p</sub>olic<sub>y</sub> can be decom<sub>p</sub>osed into its first-ste<sub>p</sub> action distribution

$$
\mu ( a ) : = \pi ( a \mid s ) \in \Delta ( \mathcal { V } )
$$

and continuation <sub>p</sub>olicies after each child state [�, �]. Hence, b<sub>y</sub> the <sub>p</sub>rinci<sub>p</sub>le of o<sub>p</sub>timalit<sub>y,</sub>

$$
V ^ { * } ( s ) = \operatorname* { m a x } _ { \mu \in \Lambda ( \mathcal { V } ) } \left\{ \sum _ { a \in \mathcal { V } } \mu ( a ) V ^ { * } ( [ s , a ] ) - \beta \sum _ { a \in \mathcal { V } } \mu ( a ) \log \frac { \mu ( a ) } { \pi _ { \mathrm { r e f } } ( a \mid s ) } \right\} .
$$

This is a strictl<sub>y</sub> concave o<sub>p</sub>timization <sub>p</sub>roblem over the sim<sub>p</sub>lex. Introduce a La<sub>g</sub>ran<sub>g</sub>e multi<sub>p</sub>lier $\xi$ f<sub>or</sub> th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t $\begin{array} { r } { \sum _ { a } \mu ( a ) = 1 } \end{array}$ Th<sub>e</sub> L<sub>ag</sub>r<sub>a</sub>n<sub>g</sub>i<sub>a</sub>n i<sub>s</sub>

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \mu , \xi ) = \sum _ { a \in \mathcal { V } } \mu ( a ) V ^ { * } ( [ s , a ] ) - \beta \sum _ { a \in \mathcal { V } } \mu ( a ) \log \frac { \mu ( a ) } { \pi _ { \mathrm { r e f } } ( a \mid s ) } } \\ { \displaystyle ~ + \xi \left( \sum _ { a \in \mathcal { V } } \mu ( a ) - 1 \right) . } \end{array}
$$

![](images/3a1ef1be72cdf4fde6f5d34c47727e444818d649d607f9bcd2a345495e3fe92f.jpg)  
Figure 3: Potential commercialization formats for LAMA. (a) Hyperlink: the winning advertiser’s brand name in the generated response is hyperlinked to its landing page; this is the format used in our experiments. (b) Clickable Block: the entire response block links to the advertiser’s landing page. (c) Sponsored Card: an advertisement card is displayed below the generated response, a format currently adopted by platforms such as ChatGPT Go, Google AI Overviews, and Baidu AI Smart Answers.

F<sub>o</sub>r <sub>eve</sub>r<sub>y</sub> <sub>ac</sub>ti<sub>o</sub>n <sub>�</sub> <sub>w</sub>ith $\mu ( a ) > 0 ,$ th<sub>e</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> <sub>con</sub>diti<sub>on</sub> <sub>g</sub>i<sub>ves</sub>

$$
\frac { \partial \mathcal { L } } { \partial \mu ( a ) } = V ^ { \ast } ( [ s , a ] ) - \beta \left( \log \frac { \mu ( a ) } { \pi _ { \mathrm { r e f } } ( a \mid s ) } + 1 \right) + \xi = 0 .
$$

Th<sub>ere</sub>f<sub>ore</sub>

$$
\log \frac { \mu ( a ) } { \pi _ { \mathrm { r e f } } ( a \mid s ) } = \frac { V ^ { * } ( [ s , a ] ) + \xi - \beta } { \beta } ,
$$

or e<sub>q</sub>uivalentl<sub>y</sub>

$$
\mu ( a ) = \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( { \frac { V ^ { * } ( [ s , a ] ) } { \beta } } \right) \exp \left( { \frac { \xi - \beta } { \beta } } \right) .
$$

Th<sub>e</sub> fi<sub>na</sub>l <sub>exponen</sub>ti<sub>a</sub>l f<sub>ac</sub>t<sub>or</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $^ { a , }$ <sub>so</sub> n<sub>o</sub>rm<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>g</sub><sup>i</sup>ves

$$
\mu ^ { * } ( a ) = \frac { \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( V ^ { * } ( [ s , a ] ) / \beta \right) } { \sum _ { a ^ { \prime } \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a ^ { \prime } \mid s ) \exp \left( V ^ { * } ( [ s , a ^ { \prime } ] ) / \beta \right) } .
$$

This is the o<sub>p</sub>timal contin<sub>u</sub>ation <sub>p</sub>olic<sub>y</sub> $\pi ^ { * } ( a \mid s )$

It remains to com<sub>p</sub>ute the o<sub>p</sub>timal value. Let

$$
Z ( s ) : = \sum _ { a \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( \frac { V ^ { \ast } ( [ s , a ] ) } { \beta } \right) .
$$

From the ex<sub>p</sub>ression for $\mu ^ { * }$

$$
\log { \frac { \mu ^ { * } ( a ) } { \pi _ { \mathrm { r e f } } ( a \mid s ) } } = { \frac { V ^ { * } ( [ s , a ] ) } { \beta } } - \log Z ( s ) .
$$

Su<sup>b</sup>stituting t<sup>h</sup>is i<sup>d</sup>entity into t<sup>h</sup>e one-step o<sup>b</sup>jective gives

$$
\begin{array} { l } { { { \displaystyle V ^ { * } ( s ) = \sum _ { a } \mu ^ { * } ( a ) V ^ { * } ( [ s , a ] ) - \beta \sum _ { a } \mu ^ { * } ( a ) \log \frac { \mu ^ { * } ( a ) } { \pi _ { \mathrm { r e f } } ( a \mid s ) } } } } \\ { { { } } } \\ { { { = \displaystyle \sum _ { a } \mu ^ { * } ( a ) V ^ { * } ( [ s , a ] ) - \beta \sum _ { a } \mu ^ { * } ( a ) \left( \frac { V ^ { * } ( [ s , a ] ) } { \beta } - \log Z ( s ) \right) } } } \\ { { { } } } \\ { { { = \displaystyle \beta \log Z ( s ) . } } } \end{array}
$$

H<sub>e</sub>n<sub>ce</sub>

$$
V ^ { * } ( s ) = \beta \log \sum _ { \alpha \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( \frac { V ^ { * } ( [ s , a ] ) } { \beta } \right) ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>proves</sub> th<sub>e</sub> <sub>so</sub>ft B<sub>e</sub>ll<sub>man</sub> <sub>recurs</sub>i<sub>on.</sub>

Lemma 2 (Convexity and gradient of weighted log-sum– exp). Let $\beta > 0 ,$ let K be a finite index set, and let $\{ w _ { k } \} _ { k \in \mathcal K }$ be nonnegative weights, not all zero. Define

$$
F _ { \beta } ( \boldsymbol { x } ) = \beta \log \left( \sum _ { k \in \mathcal { K } } w _ { k } \exp \left( \frac { x _ { k } } { \beta } \right) \right) , \qquad \boldsymbol { x } \in \mathbb { R } ^ { \mathcal { K } } .
$$

Then $F _ { \beta }$ is convex on $\mathbb { R } ^ { \mathcal { K } }$ . Moreover, $F _ { \beta }$ is diferentiable and, for each $k \in \mathcal { K } ,$

$$
\frac { \partial F _ { \beta } ( x ) } { \partial x _ { k } } = \frac { w _ { k } \exp ( x _ { k } / \beta ) } { \sum _ { k ^ { \prime } \in \mathcal { K } } w _ { k ^ { \prime } } \exp ( x _ { k ^ { \prime } } / \beta ) } .
$$

Proof<sub>.</sub> L<sub>e</sub>t

$$
Z _ { \beta } ( x ) = \sum _ { k \in \mathcal K } w _ { k } \exp \left( \frac { x _ { k } } { \beta } \right) ,
$$

<sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
p _ { k } ( x ) = \frac { w _ { k } \exp ( x _ { k } / \beta ) } { Z _ { \beta } ( x ) } .
$$

Since the wei<sub>g</sub>hts are nonne<sub>g</sub>ative and not all zero<sub>,</sub> $Z _ { \beta } ( x ) > 0$ f<sub>or</sub> ever<sub>y</sub> �. <sup>H</sup>ence $p ( x ) = ( p _ { k } ( x ) ) _ { k \in \mathcal K }$ is a <sub>p</sub>robabilit<sub>y</sub> vector over $\mathcal { K } .$ Dir<sub>ec</sub>t dif<sub>e</sub>r<sub>e</sub>nti<sub>a</sub>ti<sub>o</sub>n <sub>g</sub>i<sub>ves</sub>

$$
{ \frac { \partial F _ { \beta } ( x ) } { \partial x _ { k } } } = \beta \cdot { \frac { 1 } { Z _ { \beta } ( x ) } } \cdot w _ { k } \cdot { \frac { 1 } { \beta } } \exp \left( { \frac { x _ { k } } { \beta } } \right) = { \mathit { p } } _ { k } ( x ) .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\nabla F _ { \beta } ( x ) = p ( x ) .
$$

We next com<sub>pu</sub>te the Hessian. For an<sub>y</sub> � $, k ^ { \prime } \in \mathcal { K } ,$

$$
\frac { \partial p _ { k } ( x ) } { \partial x _ { k ^ { \prime } } } = \frac { 1 } { \beta } p _ { k } ( x ) \left( 1 \{ k = k ^ { \prime } \} - p _ { k ^ { \prime } } ( x ) \right) .
$$

Th<sub>us</sub>

$$
\nabla ^ { 2 } F _ { \beta } ( x ) = \frac { 1 } { \beta } \left( \mathrm { d i a g } ( p ( x ) ) - p ( x ) p ( x ) ^ { \top } \right) .
$$

<sup>F</sup>or an<sub>y</sub> vector $\boldsymbol { v } \in \mathbb { R } ^ { \mathcal { K } }$

$$
v ^ { \top } \nabla ^ { 2 } F _ { \beta } ( x ) v = \frac { 1 } { \beta } \left[ \sum _ { k \in \mathcal { K } } \hat { p } _ { k } ( x ) v _ { k } ^ { 2 } - \left( \sum _ { k \in \mathcal { K } } \hat { p } _ { k } ( x ) v _ { k } \right) ^ { 2 } \right] .
$$

Algorithm 2 Practical Trainin of the Shared Re ort Model   
Require: Advertiser-Query pairs $\mathcal { D } _ { I Q }$ <sub>,</sub> Ad<sub>ver</sub>ti<sub>ser pre</sub>fi<sub>x</sub> $\{ \kappa _ { i } \} _ { i \in N } ,$   
R<sub>e</sub>f<sub>erence o</sub>li<sub>c</sub> $\pi _ { \mathrm { r e f } } ,$ Rollout bud<sub>g</sub>et �<sub>,</sub> Tem<sub>p</sub>erature $\beta$   
1<sub>:</sub> // Construct pairwise comparison data   
2: $\widehat { \mathcal { D } } _ { \mathrm { p a i r } }  \emptyset , \widehat { \mathcal { D } } _ { \mathrm { r o o t } }  \emptyset$ <sub>⊲</sub> Initi<sub>a</sub>liz<sub>e</sub> tr<sub>a</sub>inin<sub>g</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub>   
3: for each advertiser-query pair $( i , q ) \in \mathcal { D } _ { I Q }$ do   
4: for $b = 1 , \ldots , B$ do   
5: $y _ { i , q } ^ { ( b ) } \sim \pi _ { \mathrm { r e f } } ( . ~ | ~ [ \kappa _ { i } , q ] )$ <sub>⊲</sub> S<sub>am</sub> l<sub>e res onses</sub>   
6: $\begin{array} { r } { r _ { i } ( \ r _ { \mathscr { t } _ { y _ { i , q } ^ { ( b ) } } } ) \gets \mathrm { O n l i n e T e s t } ( i , q , y _ { i , q } ^ { ( b ) } ) } \end{array}$ ⊲ Estimate   
<sub>p</sub>o<sub>p</sub>ulation im<sub>p</sub>ression value from online ex<sub>p</sub>eriments   
7: end for   
8: for every $1 \leq b < b ^ { \prime } \leq B$ do <sub>⊲</sub> Full com<sub>p</sub>arison   
9: $\omega _ { i } ^ { ( b , b ^ { \prime } ) } ( q ) \gets \sigma \Big ( r _ { i } ( \ell _ { y _ { i , q } ^ { ( b ) } } ) - r _ { i } ( \ell _ { y _ { i , q } ^ { ( b ^ { \prime } ) } } ) \Big )$   
10: $\begin{array} { r } { \widehat { \mathcal { D } } _ { \mathrm { p a i r } }  \widehat { \mathcal { D } } _ { \mathrm { p a i r } } \cup \{ \big ( i , q , y _ { i , q } ^ { ( b ) } , y _ { i , q } ^ { ( b ^ { \prime } ) } , \omega _ { i } ^ { ( b , b ^ { \prime } ) } ( q ) \big ) \} } \end{array}$   
11: end for   
12: end for   
13: $/ /$ Trai<sub>n</sub> local ad<sub>v</sub>a<sub>n</sub>tages   
14: $\pi _ { \theta }  \pi _ { \mathrm { r e f } }$ <sub>⊲</sub> Initialize advertiser-shared <sub>p</sub>olic<sub>y</sub> model   
$\Bigg \lceil \sum _ { t = 0 } ^ { T _ { y } - 1 } \log \frac { \pi _ { \theta , i } ( a _ { t } \mid [ \kappa _ { i } , s _ { t } ] ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid [ \kappa _ { i } , s _ { t } ] ) }$   
15: $\Delta _ { i } ^ { \pi _ { \theta } } ( q ; y , y ^ { \prime } )  \beta$   
$\Bigg \vert \mathrm { ~ } - \sum _ { t = 0 } ^ { T _ { y ^ { \prime } } - 1 } \log \frac { \pi _ { \theta , i } ( a _ { t } ^ { \prime } \mid [ \kappa _ { i } , s _ { t } ^ { \prime } ] ) } { \pi _ { \mathrm { r e f } } ( a _ { t } ^ { \prime } \mid [ \kappa _ { i } , s _ { t } ^ { \prime } ] ) } \Bigg .$   
16: $\widehat { \mathcal { L } } ( \pi _ { \theta } ) \gets \frac { 1 } { | \widehat { \mathcal { D } } _ { \mathrm { p a i r } } | } \sum _ { ( i , q , y , y ^ { \prime } , \omega ) } \left[ { \omega \log \sigma \big ( \Delta _ { i } ^ { \pi _ { \theta } } ( q ; y , y ^ { \prime } ) \big ) } \right.$   
$\in \widehat { \mathcal { D } } _ { \mathrm { p a i r } }$   
17: ${ \boldsymbol { \theta } } ^ { \dagger } \gets$ ar<sub>g</sub> m<sup>i</sup>n<sub>�</sub> $\widehat { \mathcal { L } } ( \pi _ { \theta } )$ <sub>⊲</sub> O<sub>p</sub>timize local advanta<sub>g</sub>es   
18<sub>:</sub> // Learn root values from residual targets   
19: for each advertiser-query pair $( i , q ) \in \mathcal { D } _ { I Q }$ do   
20: for $b = 1 , \ldots , B$ do   
21: $\widetilde { y } _ { i , q } ^ { ( b ) } \sim \pi _ { \theta ^ { \dagger } , i } ( \cdot \mid [ \kappa _ { i } , q ] )$ ⊲ <sup>S</sup>am<sub>p</sub><sup>l</sup>e on-<sub>p</sub>o<sup>li</sup>c<sub>y</sub> res<sub>p</sub>onses   
22: $r _ { i } ( \ r _ { \widetilde { y } _ { i , q } ^ { ( b ) } } ) \gets \mathrm { O n l i n e T e s t } ( i , q , \widetilde { y } _ { i , q } ^ { ( b ) } )$   
23: $\widetilde v _ { i , q } ^ { ( b ) } \gets r _ { i } ( \ell _ { \widetilde { y } _ { i , q } ^ { ( b ) } } ) - \beta \sum _ { t = 0 } ^ { I _ { \widetilde { y } } - 1 } \log \frac { \pi _ { \theta ^ { \dagger } , i } ( a _ { t } \mid [ \kappa _ { i } , s _ { t } ] ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid [ \kappa _ { i } , s _ { t } ] ) }$   
24: $\widehat { \mathcal { D } } _ { \mathrm { r o o t } }  \widehat { \mathcal { D } } _ { \mathrm { r o o t } } \cup \Big \{ \big ( i , q , \widetilde { v } _ { i , q } ^ { ( b ) } \big ) \Big \}$   
25: end for   
26: end for   
27: � ← AtachValueHead $1 ( \pi _ { \theta ^ { \dagger } } )$ <sub>⊲</sub> I<sub>n</sub>iti<sub>a</sub>li<sub>ze roo</sub>t<sub>-va</sub>l<sub>ue mo</sub>d<sub>e</sub>l   
28: $\phi ^ { \dagger }  \arg \operatorname* { m i n } _ { \phi } \frac { 1 } { | \widehat { \mathcal { D } } _ { \mathrm { r o o t } } | } \sum _ { ( i , q , \widetilde { v } ) \in \widehat { \mathcal { D } } _ { \mathrm { r o o t } } } ( v _ { \phi , i } ( q ) - \widetilde { v } ) ^ { 2 }$   
29: return Shared re ort mode $\left( \pi _ { \theta ^ { \dagger } } , v _ { \phi ^ { \dagger } } \right)$

Th<sub>e</sub> t<sub>erm</sub> i<sub>ns</sub>id<sub>e</sub> th<sub>e</sub> b<sub>rac</sub>k<sub>e</sub>t<sub>s</sub> i<sub>s</sub> th<sub>e var</sub>i<sub>ance o</sub>f th<sub>e ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>e</sub> th<sub>a</sub>t t<sub>a</sub>k<sub>es va</sub>l<sub>ue</sub> ${ \boldsymbol { v } } _ { k }$ with <sub>p</sub>robabilit<sub>y</sub> $p _ { k } ( x )$ <sub>.</sub> H<sub>e</sub>n<sub>ce</sub> it i<sub>s</sub> n<sub>o</sub>nn<sub>ega</sub>ti<sub>ve.</sub> Sin<sub>ce</sub> $\beta > 0 ,$ <sub>we</sub> h<sub>ave</sub>

$$
v ^ { \top } \nabla ^ { 2 } F _ { \beta } ( x ) v \geq 0
$$

<sup>f</sup>or ever<sub>y</sub> � ${ \bf \Pi } _ { 1 } \in \mathbb { R } ^ { \mathcal { K } }$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
\nabla ^ { 2 } F _ { \beta } ( x ) \succeq 0 ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t $F _ { \beta }$ i<sub>s co</sub>n<sub>ve</sub>x<sub>.</sub>

Lemma 3 (Desirability Representation). Fix an initial context   
$q .$ Let $\widehat { V } = \big ( \widehat { V } _ { i } ( u ) \big ) _ { i \in N , u \succeq q }$ be a globally Bellman-consistent report   
ledger on the subtree rooted at �. Define   
$\widehat { z } _ { i } ( u ) : = \exp \left( \frac { \widehat { V } _ { i } ( u ) } { \beta } \right) , \ : \overline { { z } } ( u ; \widehat { V } ) : = \sum _ { j \in N } \widehat { z } _ { j } ( u ) .$   
Then the online decision rule is equivalently represented by   
$\rho _ { i } ( u ; \widehat { V } ) = \frac { \widehat { z } _ { i } ( u ) } { \overline { { z } } ( u ; \widehat { V } ) } , x _ { \widehat { V } } ( a \mid u ) = \pi _ { \mathrm { r e f } } ( a \mid u ) \frac { \overline { { z } } ( [ u , a ] ; \widehat { V } ) } { \overline { { z } } ( u ; \widehat { V } ) }$   
for every reached non-terminal descendant $u \succeq q .$ In particular, ifthe   
rollout terminates at $\ell \in S _ { T }$ , then the terminal ad allocation is   
$\lambda _ { i } ( \ell ; \widehat { V } ) = \rho _ { i } ( \ell ; \widehat { V } ) = \frac { \widehat { z } _ { i } ( \ell ) } { \overline { { z } } ( \ell ; \widehat { V } ) } .$   
Proof. Global Bellman consistenc<sub>y</sub> im<sub>p</sub>lies that<sub>,</sub> for e<sub>v</sub>er<sub>y</sub> non  
t<sub>erm</sub>i<sub>na</sub>l d<sub>escen</sub>d<sub>an</sub>t $u \succeq q ,$   
${ \widehat { z } } _ { i } ( u ) = \sum _ { a \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid u ) { \widehat { z } } _ { i } ( [ u , a ] ) .$   
H<sub>ence</sub> th<sub>e a</sub>d<sub>ver</sub>ti<sub>ser-s ec</sub>ifi<sub>c</sub> t<sub>o</sub>k<sub>en ru</sub>l<sub>e can</sub> b<sub>e rewr</sub>itt<sub>en as</sub>   
${ \widehat { \pi } } _ { i } ( a \mid u ) = \pi _ { \mathrm { r e f } } ( a \mid u ) { \frac { { \widehat { z } } _ { i } ( [ u , a ] ) } { { \widehat { z } } _ { i } ( u ) } } .$   
We <sub>p</sub>rove the re<sub>p</sub>resentation b<sub>y</sub> induction alon<sub>g</sub> the realized <sub>p</sub>ath.   
At th<sub>e</sub> r<sub>oo</sub>t<sub>,</sub> th<sub>e</sub> initi<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>g</sub>i<sub>ves</sub>   
$\rho _ { i } ( q ; \widehat { V } ) = \frac { \widehat { z } _ { i } ( q ) } { \overline { { z } } ( q ; \widehat { V } ) } .$   
S<sub>uppose</sub> th<sub>e c</sub>l<sub>a</sub>i<sub>m</sub> h<sub>o</sub>ld<sub>s a</sub>t <sub>a reac</sub>h<sub>e</sub>d <sub>non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e �.</sub> Th<sub>en</sub>   
$x _ { \widehat { V } } ( a \mid u ) = \sum _ { i \in N } \rho _ { i } ( u ; \widehat { V } ) \widehat { \pi } _ { i } ( a \mid u )$   
$= \sum _ { i \in N } \frac { \widehat { z } _ { i } ( u ) } { \overline { { z } } ( u ; \widehat { V } ) } \cdot \pi _ { \mathrm { r e f } } ( a \mid u ) \frac { \widehat { z } _ { i } ( [ u , a ] ) } { \widehat { z } _ { i } ( u ) }$   
$= \pi _ { \mathrm { r e f } } ( \boldsymbol { a } \mid \boldsymbol { u } ) \frac { \overline { { \boldsymbol { z } } } ( [ \boldsymbol { u } , \boldsymbol { a } ] ; \widehat { V } ) } { \overline { { \boldsymbol { z } } } ( \boldsymbol { u } ; \widehat { V } ) } .$   
The Ba<sub>y</sub>esian <sub>p</sub>osterior u<sub>p</sub>date then <sub>g</sub>ives   
$\begin{array} { l } { \displaystyle { \rho _ { i } ( [ u , a ] ; \widehat { V } ) = \frac { \rho _ { i } ( u ; \widehat { V } ) \widehat \pi _ { i } ( a \mid u ) } { x _ { \widehat { V } } ( a \mid u ) } } } \\ { \displaystyle { = \frac { \widehat z _ { i } ( [ u , a ] ) } { \overline { { z } } ( [ u , a ] ; \widehat { V } ) } . } } \end{array}$   
Thus the re<sub>p</sub>resentation is <sub>p</sub>reserved after ever<sub>y</sub> realized transition.   
The terminal ex<sub>p</sub>ression follows b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the same <sub>p</sub>osterior   
re<sub>p</sub>resentation at ℓ. □

Lemma 4 (Joint Allocation Probability). For every globally Bellman-consistent report ledger ${ \widehat { V } } ,$ , advertiser ${ \mathfrak { i } } \in N ,$ and terminal state $\ell \in L ( q )$ , the joint allocation mass assigned to advertiser � at terminal state ℓ is

$$
\alpha _ { i , \ell } ( \widehat { V } ) : = \operatorname* { P r } _ { x _ { \widehat { V } } } ( s _ { T } = \ell \mid q ) \lambda _ { i } ( \ell ; \widehat { V } ) = P _ { 0 } ( \ell \mid q ) \frac { \exp ( \widehat { V } _ { i } ( \ell ) / \beta ) } { \overline { { z } } ( q ; \widehat { V } ) } ,
$$

where $P _ { 0 } ( \ell \mid q )$ is the probability of reaching ℓ from � under the reference policy $\pi _ { \mathrm { r e f } }$ . Equivalently,

$$
\alpha _ { i , \ell } ( \widehat { V } ) = \frac { P _ { 0 } ( \ell \mid q ) \exp ( \widehat { V } _ { i } ( \ell ) / \beta ) } { \sum _ { \ell ^ { \prime } \in L ( q ) } P _ { 0 } ( \ell ^ { \prime } \mid q ) \sum _ { j \in N } \exp ( \widehat { V } _ { j } ( \ell ^ { \prime } ) / \beta ) } .
$$

Proof. Let the uni<sub>q</sub>ue <sub>p</sub>ath from $q = s _ { 0 }$ to $\ell = s _ { T }$ b<sub>e</sub>

$$
s _ { t + 1 } = [ s _ { t } , a _ { t } ] , t = 0 , . . . , T - 1 .
$$

B<sub>y</sub> Lemma 3<sub>,</sub>

$$
x _ { \widehat { V } } ( a _ { t } \mid s _ { t } ) = \pi _ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) \frac { \overline { { { z } } } ( s _ { t + 1 } ; \widehat { V } ) } { \overline { { { z } } } ( s _ { t } ; \widehat { V } ) } .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\begin{array} { l } { { \displaystyle { \operatorname* { P r } _ { x } } ( s _ { T } = \ell \mid q ) = \prod _ { t = 0 } ^ { T - 1 } { x _ { \widehat { V } } ( a _ { t } \mid s _ { t } ) } } } \\ { { \displaystyle ~ = \prod _ { t = 0 } ^ { T - 1 } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) \prod _ { t = 0 } ^ { T - 1 } { \overline { { { \frac { \tau } { z } ( s _ { t + 1 } ; \widehat { V } ) } } } } } } } \\ { { \displaystyle ~ = P _ { 0 } ( \ell \mid q ) \overline { { { \frac { z } { z ( q ; \widehat { V } ) } } } } . } } \end{array}
$$

Th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>equa</sub>l<sub>s</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>pos</sub>t<sub>er</sub>i<sub>or:</sub>

$$
\lambda _ { i } ( \ell ; \widehat { V } ) = \rho _ { i } ( \ell ; \widehat { V } ) = \frac { \widehat { z } _ { i } ( \ell ) } { \overline { { z } } ( \ell ; \widehat { V } ) } = \frac { \exp ( \widehat { V } _ { i } ( \ell ) / \beta ) } { \overline { { z } } ( \ell ; \widehat { V } ) } .
$$

Multi<sub>p</sub>l<sub>y</sub>in<sub>g</sub> the two identities <sub>g</sub>ives

$$
\alpha _ { i , \ell } ( \widehat { V } ) = P _ { 0 } ( \ell \mid q ) \frac { \exp ( \widehat { V } _ { i } ( \ell ) / \beta ) } { \overline { { z } } ( q ; \widehat { V } ) } .
$$

Finall<sub>y,</sub> Bellman consistenc<sub>y</sub> im<sub>p</sub>lies the leaf re<sub>p</sub>resentation

$$
\overline { { z } } ( q ; \widehat { V } ) = \sum _ { j \in N } \widehat { z } _ { j } ( q ) = \sum _ { \ell ^ { \prime } \in L ( q ) } P _ { 0 } ( \ell ^ { \prime } \mid q ) \sum _ { j \in N } \exp ( \widehat { V } _ { j } ( \ell ^ { \prime } ) / \beta ) ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t <sub>norma</sub>li<sub>ze</sub>d f<sub>orm.</sub>

Lemma 5 (Gradient Representation). Define the soft potential

$$
\begin{array} { r } { \Phi _ { q } ( \widehat { V } ) : = \beta \log \overline { { z } } ( q ; \widehat { V } ) . } \end{array}
$$

Then the joint allocation rule is the gradient of this potential with respect to terminal reported values: $\nabla \Phi _ { q } ( \widehat { V } ) = \alpha ( \widehat { V } )$ , or equivalently, for every � ∈ N and $\begin{array} { r } { \ell \in L ( q ) , { \frac { \partial \Phi _ { q } ( \widehat { V } ) } { \partial \widehat { V } _ { i } ( \ell ) } } = \alpha _ { i , \ell } ( \widehat { V } ) } \end{array}$ . Consequently, � is cyclically monotone.

P<sub>r</sub>oof<sub>.</sub> B<sub>y</sub> th<sub>e</sub> l<sub>ea</sub>f <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> i<sub>n</sub>d<sub>uce</sub>d d<sub>es</sub>i<sub>ra</sub>bilit<sub>y</sub> f<sub>unc</sub>ti<sub>ons,</sub>

$$
\begin{array} { l } { \displaystyle \overline { { z } } ( { q } ; \widehat { V } ) = \sum _ { j \in N } \widehat { z } _ { j } ( { q } ) } \\ { = \displaystyle \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) \sum _ { j \in N } \exp \left( \frac { \widehat { V } _ { j } ( \ell ) } { \beta } \right) . } \end{array}
$$

Th<sub>us</sub>

$$
\Phi _ { q } ( \widehat { V } ) = \beta \log \left[ \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) \sum _ { j \in N } \exp \left( \frac { \widehat { V } _ { j } ( \ell ) } { \beta } \right) \right] .
$$

Diferentiatin<sub>g</sub> directl<sub>y</sub> with res<sub>p</sub>ect to $\widehat { V } _ { i } ( \ell )$ <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { l } { \displaystyle \frac { \partial \Phi _ { q } ( \widehat { V } ) } { \partial \widehat { V } _ { i } ( \ell ) } = \beta \frac { P _ { 0 } ( \ell \mid q ) \frac { 1 } { \beta } \exp \Big ( \widehat { V } _ { i } ( \ell ) / \beta \Big ) } { \overline { { z } } ( q ; \widehat { V } ) } } \\ { = P _ { 0 } ( \ell \mid q ) \frac { \exp \big ( \widehat { V } _ { i } ( \ell ) / \beta \big ) } { \overline { { z } } ( q ; \widehat { V } ) } } \\ { = \alpha _ { i , \ell } ( \widehat { V } ) , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t <sub>equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> L<sub>emma</sub> 4<sub>.</sub>

Finall<sub>y,</sub> $\Phi _ { q }$ is a <sub>p</sub>ositive scalar multi<sub>p</sub>le of a lo<sub>g</sub>-sum-ex<sub>p</sub> function<sub>,</sub> and b<sub>y</sub> Lemma 2<sub>,</sub> it is therefore con<sub>v</sub>ex. Since $\alpha = \nabla \Phi _ { q } ,$ <sub>,</sub> th<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on ru</sub>l<sub>e �</sub> i<sub>s cyc</sub>li<sub>ca</sub>ll<sub>y mono</sub>t<sub>one.</sub> □

Proposition 2 (Direct implementability of LAMA). For any prefix $q ,$ the direct latent-mixture allocation rule induced by globally Bellman-consistent reports is implementable with quasi-linear payments. Therefore truthful reporting $\widehat { V } _ { i } = V _ { i } ^ { * }$ is DSIC and IR.

Proof of Proposition 2<sub>.</sub> Fix <sub>a</sub>d<sub>ve</sub>rti<sub>se</sub>r �<sub>,</sub> tr<sub>ue</sub> t<sub>ype</sub> $r _ { i } ,$ <sup>o</sup>pp<sup>o-</sup> <sup>nent</sup> <sup>re</sup>p<sup>orts</sup> $\widehat { V } _ { - i }$ <sub>,</sub> <sub>an</sub>d <sub>an</sub> <sub>ar</sub>bit<sub>rary</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> B<sub>e</sub>ll<sub>man-cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>re-</sub> p<sup>ort</sup> ${ \widehat { V } } _ { i } .$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> th<sub>e ou</sub>t<sub>s</sub>id<sub>e-op</sub>ti<sub>on o</sub>f<sub>se</sub>t

$$
\Phi _ { - i , q } ^ { 0 } ( \widehat { V } _ { - i } ) : = \beta \log \left( 1 + \sum _ { j \neq i } \widehat { z } _ { j } ( q ) \right) .
$$

The direct <sub>p</sub>a<sub>y</sub>ment corres<sub>p</sub>ondin<sub>g</sub> to the <sub>p</sub>athwise LAMA <sub>p</sub>a<sub>y</sub>ment <sub>ru</sub>l<sub>e</sub> i<sub>s</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l<sub>-se</sub>ttl<sub>e</sub>d <sub>c</sub>h<sub>arge</sub>

$$
p _ { i } ( \ell ; \widehat { V } ) : = \lambda _ { i } ( \ell ; \widehat { V } ) \widehat { V } _ { i } ( \ell ) - \Phi _ { q } ( \widehat { V } ) + \Phi _ { - i , q } ^ { 0 } ( \widehat { V } _ { - i } ) .
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> th<sub>e sequen</sub>ti<sub>a</sub>l <sub>en</sub>t<sub>ry</sub> f<sub>ee equa</sub>l<sub>s</sub> $\rho _ { i } ( q ; \widehat { V } ) \widehat { V } _ { i } ( q ) - \Phi _ { q } ( \widehat { V } ) +$ $\Phi _ { - i , q } ^ { 0 } ( \widehat { V } _ { - i } )$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>con</sub>ti<sub>nua</sub>ti<sub>on-va</sub>l<sub>ue</sub> <sub>c</sub>h<sub>arges</sub> t<sub>e</sub>l<sub>e-</sub> <sup>sco</sup>p<sup>e</sup> <sup>to</sup> $\lambda _ { i } ( \ell ; \widehat { V } ) \widehat { V } _ { i } ( \ell ) - \lambda _ { i } ( q ; \widehat { V } ) \widehat { V } _ { i } ( q )$

Ad<sub>v</sub>ertiser �’s ex<sub>p</sub>ected <sub>u</sub>tilit<sub>y</sub> <sub>u</sub>nder re<sub>p</sub>ort $\widehat { V } _ { i }$ i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub>

$$
\begin{array} { l } { { U _ { i } ( q ; r _ { i } , \widehat { V } _ { i } , \widehat { V } _ { - i } ) = \displaystyle \sum _ { \ell \in L ( q ) } \operatorname* { P r } _ { x \widehat { V } } ( s _ { T } = \ell \mid q ) \lambda _ { i } ( \ell ; \widehat { V } ) \big ( r _ { i } ( \ell ) - \widehat { V } _ { i } ( \ell ) \big ) } } \\ { { \displaystyle \quad \quad + \Phi _ { q } ( \widehat { V } ) - \Phi _ { - i , q } ^ { 0 } ( \widehat { V } _ { - i } ) } } \\ { { \displaystyle \quad = \sum _ { \ell \in L ( q ) } \alpha _ { i , \ell } ( \widehat { V } ) \big ( r _ { i } ( \ell ) - \widehat { V } _ { i } ( \ell ) \big ) + \Phi _ { q } ( \widehat { V } ) - \Phi _ { - i , q } ^ { 0 } ( \widehat { V } _ { - i } ) . } } \end{array}
$$

B<sub>y</sub> Lemma 5<sub>,</sub>

$$
\alpha _ { i , \ell } ( \widehat { V } ) = \frac { \partial \Phi _ { q } ( \widehat { V } ) } { \partial \widehat { V } _ { i } ( \ell ) } .
$$

H<sub>ence,</sub> <sub>w</sub>ith th<sub>e</sub> i<sub>nner</sub> <sub>pro</sub>d<sub>uc</sub>t t<sub>a</sub>k<sub>en</sub> <sub>over</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es</sub> $L ( q )$

$$
\begin{array} { r l } & { U _ { i } ( q ; r _ { i } , \widehat { V } _ { i } , \widehat { V } _ { - i } ) = \Phi _ { q } ( \widehat { V } _ { i } , \widehat { V } _ { - i } ) + \Big \langle \nabla _ { i } \Phi _ { q } \big ( \widehat { V } _ { i } , \widehat { V } _ { - i } \big ) , r _ { i } - \widehat { V } _ { i } \big | _ { L ( q ) } \Big \rangle } \\ & { \qquad - \Phi _ { - i , q } ^ { 0 } \big ( \widehat { V } _ { - i } \big ) . } \end{array}
$$

Sin<sub>ce</sub> $\Phi _ { q }$ i<sub>s</sub> <sub>convex</sub> i<sub>n</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> �’<sub>s</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>repor</sub>t<sub>e</sub>d <sub>va</sub>l<sub>ues</sub> <sub>w</sub>h<sub>en</sub> $\widehat { V } _ { - i }$ i<sub>s</sub> fi<sub>xe</sub>d<sub>, an</sub>d <sub>s</sub>i<sub>nce</sub> th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l B<sub>e</sub>ll<sub>man-cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>repor</sub>t $V _ { i } ^ { * }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $V _ { i } ^ { * } ( \ell ) = r _ { i } ( \ell )$ <sup>f</sup>or ever<sub>y</sub> $\ell \in L ( q )$

$$
\Phi _ { q } ( V _ { i } ^ { * } , \widehat { V } _ { - i } ) \geq \Phi _ { q } ( \widehat { V } _ { i } , \widehat { V } _ { - i } ) + \left. \nabla _ { i } \Phi _ { q } ( \widehat { V } _ { i } , \widehat { V } _ { - i } ) , r _ { i } - \widehat { V } _ { i } | _ { L ( q ) } \right. .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
U _ { i } ( q ; r _ { i } , \widehat { V } _ { i } , \widehat { V } _ { - i } ) \leq \Phi _ { q } ( V _ { i } ^ { * } , \widehat { V } _ { - i } ) - \Phi _ { - i , q } ^ { 0 } ( \widehat { V } _ { - i } ) .
$$

Wh<sub>en</sub> $\widehat { V } _ { i } = V _ { i } ^ { * }$ <sub>,</sub> the ine<sub>q</sub>ualit<sub>y</sub> is ti<sub>g</sub>ht<sub>,</sub> so truthful re<sub>p</sub>ortin<sub>g</sub> is a <sup>d</sup>om<sup>i</sup>nant strate<sub>gy</sub>.

Th<sub>e same argumen</sub>t <sub>a</sub>l<sub>so g</sub>i<sub>ves s</sub>t<sub>r</sub>i<sub>c</sub>t d<sub>om</sub>i<sub>nance un</sub>d<sub>er</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>-</sub> s<sub>upp</sub>ort conditions. Since $| N | \ge 2$ <sub>an</sub>d $P _ { 0 } ( \ell \mid q ) ~ > ~ 0$ <sup>f</sup>or ever<sub>y</sub> $\ell \in L ( q )$ <sub>.</sub> Fixin<sub>g</sub> ${ \widehat { V } } _ { - i } ,$ write

$$
B _ { - i } ( \widehat { V } _ { - i } ) : = \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) \sum _ { j \neq i } \exp ( \widehat { V } _ { j } ( \ell ) / \beta ) .
$$

Th<sub>en</sub> $B _ { - i } ( \widehat { V } _ { - i } ) ~ > ~ 0$ <sub>,</sub> <sub>an</sub>d <sub>as</sub> <sub>a</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>o</sub>f <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> �’<sub>s</sub> t<sub>erm</sub>i<sub>na</sub>l <sup>re</sup>p<sup>ort</sup> $y = ( y _ { \ell } ) _ { \ell \in L ( q ) }$

$$
\Phi _ { q } ( y , \widehat { V } _ { - i } ) = \beta \log \Biggl ( B _ { - i } ( \widehat { V } _ { - i } ) + \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) \exp ( y _ { \ell } / \beta ) \Biggr ) ,
$$

Follo<sub>w</sub>in<sub>g</sub> the <sub>p</sub>roof of Lemma 2<sub>,</sub> its Hessian is $\begin{array} { r } { { \frac { 1 } { \beta } } ( \mathrm { d i a g } ( p ) - p p ^ { \top } ) } \end{array}$ <sub>w</sub>h<sub>ere</sub> $\begin{array} { r } { \dot { p } _ { \ell } = \frac { P _ { 0 } ( \ell \vert q ) \exp ( y _ { \ell } / \beta ) } { B _ { - i } ( \widehat { V } _ { - i } ) + \sum _ { \ell ^ { \prime } } P _ { 0 } ( \ell ^ { \prime } \vert q ) \exp ( y _ { \ell ^ { \prime } } / \beta ) } } \end{array}$ . <sup>F</sup>or an<sub>y</sub> non-zero vector $\boldsymbol { z } = ( \boldsymbol { z } _ { \ell } ) _ { \ell \in L ( \boldsymbol { q } ) }$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $\begin{array} { r } { z ^ { \top } ( \mathrm { d i a g } ( \boldsymbol { \hat { p } } ) - \boldsymbol { \hat { p } } \boldsymbol { p } ^ { \top } ) \boldsymbol { z } = \sum _ { \ell } \mathcal { P } \ell \boldsymbol { z } _ { \ell } ^ { 2 } - ( \sum _ { \ell } \mathcal { P } \ell \boldsymbol { z } _ { \ell } ) ^ { 2 } } \end{array}$ Let $\begin{array} { r } { s = \sum _ { \ell } \bar { p _ { \ell } } < 1 , \bar { p } _ { \ell } = \frac { \bar { p } _ { \ell } } { s } } \end{array}$ <sub>,</sub> th<sub>en</sub>

$$
\begin{array} { l } { { \displaystyle \sum _ { \ell } \bar { p } _ { \ell } z _ { \ell } ^ { 2 } - \left( \sum _ { \ell } \bar { p } _ { \ell } z _ { \ell } \right) ^ { 2 } = s \sum _ { \ell } \bar { p } _ { \ell } z _ { \ell } ^ { 2 } - s ^ { 2 } \left( \sum _ { \ell } \bar { p } _ { \ell } z _ { \ell } \right) ^ { 2 } } } \\ { { \displaystyle \vphantom { \sum _ { \ell } \bar { p } _ { \ell } } } } \\ { { \displaystyle \vphantom { \sum _ { \ell } \bar { p } _ { \ell } } } } \end{array}
$$

Th<sub>us</sub> thi<sub>s</sub> H<sub>ess</sub>i<sub>a</sub>n i<sub>s</sub> <sub>pos</sub>iti<sub>ve</sub> d<sub>e</sub>finit<sub>e.</sub> H<sub>e</sub>n<sub>ce</sub> th<sub>e</sub> <sub>co</sub>n<sub>ve</sub>xit<sub>y</sub> ine<sub>q</sub>ualit<sub>y</sub> above is strict whenever $\widehat { V } _ { i } | _ { L ( q ) } \neq r _ { i }$ <sub>.</sub> Si<sub>nce g</sub>l<sub>o</sub>b<sub>a</sub>l B<sub>e</sub>ll<sub>man</sub> consistenc<sub>y</sub> uni<sub>q</sub>uel<sub>y</sub> determines $V ^ { * }$ f<sub>rom</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>va</sub>l<sub>ues,</sub> t<sub>ru</sub>th<sub>-</sub> ful re<sub>p</sub>ortin<sub>g</sub> strictl<sub>y</sub> dominates ever<sub>y</sub> distinct <sub>g</sub>loball<sub>y</sub> Bellmancons<sup>i</sup>stent re<sub>p</sub>ort.

Finall<sub>y,</sub> truthful utilit<sub>y</sub> is

$$
U _ { i } ( q ; r _ { i } , V _ { i } ^ { * } , \widehat { V } _ { - i } ) = \Phi _ { q } ( V _ { i } ^ { * } , \widehat { V } _ { - i } ) - \Phi _ { - i , q } ^ { 0 } ( \widehat { V } _ { - i } ) .
$$

B<sub>ecause</sub> $r _ { i } ( \ell ) \geq 0 _ { : }$ <sub>,</sub> Bellman consistenc<sub>y</sub> <sub>g</sub>ives

$$
z _ { i } ^ { * } ( q ) = \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) \exp ( r _ { i } ( \ell ) / \beta ) \geq 1 .
$$

Th<sub>us</sub>

$$
\overline { { z } } ( q ; V _ { i } ^ { * } , \widehat { V } _ { - i } ) = z _ { i } ^ { * } ( q ) + \sum _ { j \neq i } \widehat { z } _ { j } ( q ) \geq 1 + \sum _ { j \neq i } \widehat { z } _ { j } ( q ) .
$$

Sin<sub>ce</sub> $\beta \log ( \cdot )$ i<sub>s</sub> in<sub>c</sub>r<sub>eas</sub>in<sub>g,</sub>

$$
\Phi _ { q } ( V _ { i } ^ { * } , \widehat { V } _ { - i } ) \geq \Phi _ { - i , q } ^ { 0 } ( \widehat { V } _ { - i } ) ,
$$

and truthful ex<sub>p</sub>ected utilit<sub>y</sub> is nonne<sub>g</sub>ative.

Lemma 6 (Subtree patching). Fix a reachable platform history ℎ with current non-terminal state �, after advertiser � has reported truthfully before ℎ. Every feasible continuation deviation $\boldsymbol { \sigma } _ { i } ^ { \prime }$ from ℎ can be embedded in a globally Bellman-consistent direct report $\widetilde { V _ { i } }$ on the full tree rooted at �: the report $\widetilde { V _ { i } }$ agrees with $V _ { i } ^ { * }$ outside the subtree rooted at � and reproduces $\boldsymbol { \sigma } _ { i } ^ { \prime }$ inside that subtree. Holding opponents fixed, conditional on $h ,$ the patched report induces the same continuation token distribution, terminal allocation, and future payments as $\sigma _ { i } ^ { \prime }$

Proof of Lemma 6<sub>.</sub> F<sub>o</sub>r <sub>eve</sub>r<sub>y</sub> n<sub>o</sub>n-t<sub>e</sub>rmin<sub>a</sub>l<sub>�</sub> $\succeq s ,$ l<sub>e</sub>t $\widehat { V } _ { i } ^ { \sigma ^ { \prime } } ( [ u , a ] )$ b<sub>e</sub> th<sub>e</sub> <sub>�-componen</sub>t <sub>su</sub>b<sub>m</sub>itt<sub>e</sub>d b<sub>y</sub> $\sigma _ { i } ^ { \prime }$ at $u ,$ <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>a</sub>t <sub>�</sub> b<sub>y</sub> its soft Bellman a<sub>gg</sub>re<sub>g</sub>ate. Feasibilit<sub>y</sub> im<sub>p</sub>lies that these val-<sub>ues are</sub> B<sub>e</sub>ll<sub>man-cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>on</sub> th<sub>e su</sub>bt<sub>ree roo</sub>t<sub>e</sub>d <sub>a</sub>t $s . \mathrm { I f } s \neq q ,$ it also im<sub>p</sub>lies $\widehat { V } _ { i } ^ { \sigma ^ { \prime } } ( s ) = V _ { i } ^ { * } ( s )$ <sub>,</sub> b<sub>ecause</sub> th<sub>e</sub> l<sub>a</sub>tt<sub>er</sub> i<sub>s</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> <sub>s</sub>t<sub>ore</sub>d i<sub>n</sub> th<sub>e</sub> l<sub>e</sub>d<sub>ger a</sub>ft<sub>er</sub> th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>pre</sub>fi<sub>x.</sub>

N<sub>ow</sub> <sub>pa</sub>t<sub>c</sub>h thi<sub>s</sub> <sub>su</sub>bt<sub>ree</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l di<sub>rec</sub>t <sub>repor</sub>t<sub>:</sub>

$$
\widetilde { V } _ { i } ( u ) : = \left\{ \begin{array} { l l } { \widehat { V } _ { i } ^ { \sigma ^ { \prime } } ( u ) , } & { u \succeq s , } \\ { V _ { i } ^ { * } ( u ) , } & { u \not \equiv s . } \end{array} \right.
$$

Lemma 1 <sub>g</sub>ives Bellman consistenc<sub>y</sub> of $V _ { i } ^ { * }$ <sub>.</sub> Wh<sub>en</sub> $s \neq q ,$ the e<sub>q</sub>ualit<sub>y</sub> <sub>a</sub>t <sub>�</sub> l<sub>eaves every ances</sub>t<sub>or</sub>’<sub>s</sub> B<sub>e</sub>ll<sub>man equa</sub>ti<sub>on unc</sub>h<sub>ange</sub>d<sub>; w</sub>h<sub>en</sub> $s =$ <sub>�,</sub> there are no ancestors. Feasibilit<sub>y</sub> <sub>g</sub>ives ever<sub>y</sub> Bellman e<sub>q</sub>uation <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> <sub>pa</sub>t<sub>c</sub>h<sub>e</sub>d <sub>su</sub>bt<sub>ree.</sub> H<sub>ence</sub> $\dot { \widetilde { V _ { i } } }$ i<sub>s</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> B<sub>e</sub>ll<sub>man-cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>on</sub> th<sub>e</sub> t<sub>ree</sub> <sub>roo</sub>t<sub>e</sub>d <sub>a</sub>t <sub>�.</sub>

B<sub>e</sub>f<sub>ore</sub> <sub>reac</sub>hi<sub>ng</sub> <sub>�,</sub> th<sub>e</sub> <sub>pa</sub>t<sub>c</sub>h<sub>e</sub>d <sub>an</sub>d t<sub>ru</sub>thf<sub>u</sub>l l<sub>e</sub>d<sub>gers</sub> <sub>co</sub>i<sub>nc</sub>id<sub>e.</sub> F<sub>rom � onwar</sub>d<sub>,</sub> th<sub>e pa</sub>t<sub>c</sub>h<sub>e</sub>d <sub>c</sub>hild <sub>va</sub>l<sub>ues are exac</sub>tl<sub>y</sub> th<sub>ose su</sub>b<sub>m</sub>itt<sub>e</sub>d <sup>b</sup><sub>y</sub> $\sigma _ { i } ^ { \prime }$ . Therefore<sub>,</sub> a<sub>g</sub>ainst fixed o<sub>pp</sub>onent re<sub>p</sub>orts<sub>,</sub> the <sub>p</sub>osterior at ℎ is unchan<sub>g</sub>ed<sub>,</sub> and Lemma 3 <sub>g</sub>ives the same continuation token rule and terminal allocation. The instantaneous <sub>p</sub>a<sub>y</sub>ment rule then <sub>g</sub>ives the same <sub>p</sub>a<sub>y</sub>ment on e<sub>v</sub>er<sub>y</sub> contin<sub>u</sub>ation transition. This <sub>p</sub>ro<sub>v</sub>es th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>me</sub>d <sub>con</sub>diti<sub>ona</sub>l <sub>equ</sub>i<sub>va</sub>l<sub>ence.</sub> □

P<sub>roof of</sub> T<sub>heorem</sub> 1<sub>.</sub> L<sub>emma</sub> 1 fi<sub>rs</sub>t <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>c</sub>hild<sub>-va</sub>l<sub>ue</sub> <sub>repor</sub>t<sub>s</sub> <sub>are</sub> f<sub>eas</sub>ibl<sub>e.</sub> Fi<sub>x</sub> <sub>a</sub>d<sub>ver</sub>ti<sub>ser</sub> �<sub>,</sub> <sub>opponen</sub>t<sub>s</sub>’ f<sub>eas</sub>ibl<sub>e</sub> strate<sub>g</sub><sup>i</sup>es $\sigma _ { - i } .$ <sub>, an</sub>d <sub>a reac</sub>h<sub>a</sub>bl<sub>e</sub> hi<sub>s</sub>t<sub>ory</sub> ℎ <sub>w</sub>ith <sub>curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s ,$ <sub>a</sub>ft<sub>er</sub> � h<sub>as repor</sub>t<sub>e</sub>d t<sub>ru</sub>thf<sub>u</sub>ll<sub>y</sub> b<sub>e</sub>f<sub>ore</sub> ℎ<sub>.</sub> R<sub>ecurs</sub>i<sub>ve</sub>l<sub>y recor</sub>di<sub>ng</sub> th<sub>e c</sub>hild <sub>va</sub>l <sub>ues</sub> <sub>su</sub>b<sub>m</sub>itt<sub>e</sub>d b<sub>y</sub> <sub>eac</sub>h <sub>opponen</sub>t <sub>g</sub>i<sub>ves</sub> <sub>a</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> B<sub>e</sub>ll<sub>man-cons</sub>i<sub>s</sub>t<sub>en</sub>t di<sub>rec</sub>t l<sub>e</sub>d<sub>ger</sub> $\widehat { V } _ { - i }$ . B<sub>y</sub> Lemma 3 and the telesco<sub>p</sub>in<sub>g</sub> <sub>p</sub>a<sub>y</sub>ment rule<sub>,</sub> the se<sub>q</sub>uential mechanism is <sub>p</sub>athwise e<sub>q</sub>uivalent to the direct im-<sub>p</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on un</sub>d<sub>er</sub> th<sub>ese</sub> l<sub>e</sub>d<sub>gers.</sub>

Let $\boldsymbol { \sigma } _ { i } ^ { \prime }$ be an<sub>y</sub> feasible contin<sub>u</sub>ation de<sub>v</sub>iation. B<sub>y</sub> Lemma $^ { 6 , }$ it i<sub>n</sub>d<sub>uces a g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> B<sub>e</sub>ll<sub>man-cons</sub>i<sub>s</sub>t<sub>en</sub>t di<sub>rec</sub>t <sub>repor</sub>t $\widetilde { V _ { i } }$ th<sub>a</sub>t dif<sub>ers</sub> f<sub>rom</sub> $V _ { i } ^ { * }$ <sub>on</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>su</sub>bt<sub>ree</sub> <sub>roo</sub>t<sub>e</sub>d <sub>a</sub>t <sub>�.</sub> P<sub>ropos</sub>iti<sub>on</sub> 2 th<sub>ere</sub>f<sub>ore</sub> im<sub>p</sub>lies

$$
\begin{array} { r l } & { 0 \leq U _ { i } ( q ; r _ { i } , V _ { i } ^ { * } , \widehat { V } _ { - i } ) - U _ { i } ( q ; r _ { i } , \widetilde { V } _ { i } , \widehat { V } _ { - i } ) } \\ & { \quad = \operatorname* { P r } ( h ) \left[ U _ { i } ^ { \Gamma } ( h ; r _ { i } , \sigma _ { i } ^ { \mathrm { t r } } , \sigma _ { - i } ; r _ { - i } ) - U _ { i } ^ { \Gamma } ( h ; r _ { i } , \sigma _ { i } ^ { \prime } , \sigma _ { - i } ; r _ { - i } ) \right] . } \end{array}
$$

H<sub>e</sub>r<sub>e</sub> $\mathrm { P r } ( h )$ is the common <sub>p</sub>robabilit<sub>y</sub> of reachin<sub>g</sub> ℎ. The e<sub>q</sub>ualit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> b<sub>ecause</sub> th<sub>e</sub> t<sub>wo</sub> di<sub>rec</sub>t <sub>repor</sub>t<sub>s</sub> <sub>co</sub>i<sub>nc</sub>id<sub>e</sub> b<sub>e</sub>f<sub>ore</sub> ℎ <sub>an</sub>d <sub>on</sub> <sub>every</sub> branch not <sub>p</sub>assin<sub>g</sub> throu<sub>g</sub>h <sub>�; p</sub>ast <sub>p</sub>a<sub>y</sub>ments cancel<sub>,</sub> and conditional <sub>on</sub> ℎ th<sub>e pa</sub>t<sub>c</sub>h<sub>e</sub>d <sub>repor</sub>t <sub>repro</sub>d<sub>uces</sub> $\boldsymbol { \sigma } _ { i } ^ { \prime }$ <sub>.</sub> Si<sub>nce</sub> ℎ i<sub>s reac</sub>h<sub>a</sub>bl<sub>e,</sub> $\operatorname* { P r } ( h ) >$ $0 ,$ so the bracketed continuation-utilit<sub>y</sub> diference is nonne<sub>g</sub>ative. Thi<sub>s p</sub>r<sub>oves</sub> M<sub>a</sub>rk<sub>ov</sub> DSIC<sub>.</sub>

Fi<sub>na</sub>ll<sub>y,</sub> f<sub>rom</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>query,</sub> th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>sequen</sub>ti<sub>a</sub>l <sub>ou</sub>t<sub>come</sub> and <sub>p</sub>a<sub>y</sub>ments are <sub>p</sub>athwise e<sub>q</sub>uivalent to the direct re<sub>p</sub>ort $( V _ { i } ^ { * } , \widehat { V } _ { - i } )$ The IR <sub>p</sub>art of Pro<sub>p</sub>osition 2 therefore <sub>g</sub>ives $U _ { i } ^ { \Gamma } ( q ; r _ { i } , \bar { \sigma } _ { i } ^ { \mathrm { t r } } , \sigma _ { - i } ; \stackrel { . } { r _ { - i } } ) \geq$ 0<sub>,</sub> <sub>p</sub>ro<sub>v</sub>in<sub>g</sub> IR. □

Proof of Theore<sub>m</sub> 2<sub>.</sub> F<sub>or every</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $\ell \in L ( q )$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
G _ { \beta } ( \ell ) : = \beta \log \left( \sum _ { i \in N } \exp \left( \frac { r _ { i } ( \ell ) } { \beta } \right) \right) .
$$

W<sub>e</sub> fi<sub>rs</sub>t <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> i<sub>n</sub>d<sub>uce</sub>d b<sub>y</sub> LAMA<sub>.</sub> U<sub>n</sub>d<sub>er</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>repor</sub>ti<sub>ng,</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l b<sub>oun</sub>d<sub>ary</sub> <sub>con</sub>diti<sub>on</sub> <sub>o</sub>f <sub>eac</sub>h <sub>a</sub>d<sub>ver</sub>ti<sub>ser-spec</sub>ifi<sub>c</sub> <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on</sub> i<sub>s</sub>

$$
V _ { i } ^ { * } ( \ell ) = r _ { i } ( \ell ) .
$$

Th<sub>ere</sub>f<sub>ore,</sub> b<sub>y</sub> L<sub>emma</sub> 3<sub>,</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>pos</sub>t<sub>er</sub>i<sub>or, an</sub>d h<sub>ence</sub> th<sub>e</sub> t<sub>er-</sub> <sub>m</sub>i<sub>na</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> i<sub>s</sub>

$$
\lambda _ { i } ^ { \beta } ( \ell ) = \frac { \exp ( V _ { i } ^ { \ast } ( \ell ) / \beta ) } { \sum _ { j \in { \cal N } } \exp ( V _ { j } ^ { \ast } ( \ell ) / \beta ) } = \frac { \exp ( r _ { i } ( \ell ) / \beta ) } { \sum _ { j \in { \cal N } } \exp ( r _ { j } ( \ell ) / \beta ) } .
$$

This allocation is <sub>p</sub>recisel<sub>y</sub> the o<sub>p</sub>timizer of the entro<sub>py</sub>-re<sub>g</sub>ularized t<sub>erm</sub>i<sub>na</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on pro</sub>bl<sub>em.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>, re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e un</sub>if<sub>orm</sub> di<sub>s</sub>t<sub>r</sub>i<sub>-</sub> b<sub>u</sub>ti<sub>on on</sub> $N ,$

$$
\mathbb { D } _ { \mathrm { K L } } \big ( \lambda \| U ( N ) \big ) = \log | N | - \mathcal { H } ( \lambda ) ,
$$

<sub>an</sub>d h<sub>ence</sub>

$$
\sum _ { i \in N } \lambda _ { i } r _ { i } ( \ell ) + \beta \mathcal { H } ( \lambda ) = \sum _ { i \in N } \lambda _ { i } r _ { i } ( \ell ) - \beta \mathbb { D } _ { \mathrm { K L } } \big ( \lambda \| U ( N ) \big ) + \beta \log | N | .
$$

T<sup>h</sup>us t<sup>h</sup>e entropy-regu<sup>l</sup>arize<sup>d</sup> o<sup>b</sup>jective <sup>d</sup>i<sup>f</sup>ers <sup>b</sup>y t<sup>h</sup>e constant � log $| N |$ from the KL-re<sub>g</sub>ularized one-ste<sub>p</sub> <sub>p</sub>roblem in Lemma 1<sub>,</sub> with action set N and terminal reward $r _ { i } ( \ell )$ for action �. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> th<sub>e</sub> l<sub>emma an</sub>d <sub>a</sub>ddi<sub>ng</sub> th<sub>e same cons</sub>t<sub>an</sub>t t<sub>o</sub> it<sub>s op</sub>ti<sub>ma</sub>l <sub>va</sub>l<sub>ue g</sub>i<sub>ves</sub>

$$
\begin{array} { l } { \displaystyle G _ { \beta } ( \boldsymbol { \ell } ) = \operatorname* { m a x } _ { \lambda \in \Delta ( N ) } \left\{ \sum _ { i \in N } \lambda _ { i } r _ { i } ( \boldsymbol { \ell } ) + \beta \mathcal { H } ( \lambda ) \right\} , } \\ { \displaystyle G _ { \beta } ( \boldsymbol { \ell } ) = \sum _ { i \in N } \lambda _ { i } ^ { \beta } ( \boldsymbol { \ell } ) r _ { i } ( \boldsymbol { \ell } ) + \beta \mathcal { H } ( \lambda ^ { \beta } ( \boldsymbol { \ell } ) ) . } \end{array}\tag{1}
$$

The maximizer is <sub>u</sub>ni<sub>qu</sub>e beca<sub>u</sub>se Shannon entro<sub>py</sub> is strictl<sub>y</sub> con-<sub>cave</sub> <sub>on</sub> th<sub>e</sub> <sub>s</sub>i<sub>mp</sub>l<sub>ex.</sub>

W<sub>e nex</sub>t <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> th<sub>e</sub> t<sub>o</sub>k<sub>en po</sub>li<sub>cy.</sub> D<sub>e</sub>fi<sub>ne</sub>

$$
W _ { \beta } ( u ) : = \beta \log \left( \sum _ { i \in N } \exp \left( \frac { V _ { i } ^ { * } ( u ) } { \beta } \right) \right) .
$$

At <sub>a</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $\ell ,$ the terminal boundar<sub>y</sub> condition <sub>g</sub>ives

$$
W _ { \beta } ( \ell ) = G _ { \beta } ( \ell ) .
$$

M<sub>o</sub>r<sub>eove</sub>r<sub>,</sub> L<sub>e</sub>mm<sub>a</sub> 1 im<sub>p</sub>li<sub>es,</sub> <sub>a</sub>t <sub>eve</sub>r<sub>y</sub> n<sub>o</sub>n-t<sub>e</sub>rmin<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>�,</sub>

$$
\exp \left( \frac { W _ { \beta } ( u ) } { \beta } \right) = \sum _ { a \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid u ) \exp \left( \frac { W _ { \beta } ( [ u , a ] ) } { \beta } \right) .\tag{2}
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> thi<sub>s</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>y</sub> <sub>exponen</sub>ti<sub>a</sub>ti<sub>ng</sub> <sub>eac</sub>h <sub>a</sub>d<sub>ver</sub>ti<sub>ser-spec</sub>ifi<sub>c</sub> <sub>so</sub>ft Bellman recursion and then summin<sub>g</sub> over advertisers. B<sub>y</sub> Lemma 3<sub>,</sub> th<sub>e</sub> tr<sub>u</sub>thf<sub>u</sub>l LAMA t<sub>o</sub>k<sub>e</sub>n <sub>po</sub>li<sub>cy</sub> i<sub>s</sub>

$$
x ^ { \beta } ( a \mid u ) = \frac { \pi _ { \mathrm { r e f } } ( a \mid u ) \exp ( W _ { \beta } ( [ u , a ] ) / \beta ) } { \sum _ { a ^ { \prime } \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a ^ { \prime } \mid u ) \exp ( W _ { \beta } ( [ u , a ^ { \prime } ] ) / \beta ) } .
$$

To<sub>g</sub>ether with the terminal boundar<sub>y</sub> condition, E<sub>q</sub>uation (2) identifi<sub>es</sub> $W _ { \beta }$ <sub>as</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>so</sub>ft <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on</sub> f<sub>or</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d $G _ { \beta } .$ H<sub>e</sub>n<sub>ce</sub> L<sub>e</sub>mm<sub>a</sub> 1 <sub>g</sub>i<sub>ves</sub>

$$
x ^ { \beta } \in \arg \operatorname* { m a x } _ { x } \mathbb { E } _ { \tau \sim x ( \cdot \vert q ) } \left[ G _ { \beta } ( \ell ( \tau ) ) - \beta \sum _ { t = 0 } ^ { T _ { \tau } - 1 } \log \frac { x ( a _ { t } \mid s _ { t } ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } \right] .\tag{3}
$$

Combinin<sub>g</sub> the terminal o<sub>p</sub>timization with the outer token-control <sub>pro</sub>bl<sub>em</sub> <sub>proves</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> f<sub>or</sub> <sub>any</sub> fi<sub>xe</sub>d t<sub>o</sub>k<sub>en</sub> <sub>po</sub>li<sub>cy</sub> <sub>�,</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ce o</sub>f �(�) $\lambda ( \tau )$ is separa<sup>bl</sup>e across termina<sup>l</sup> trajectories. By

E<sub>q</sub>uation (1),

$$
\begin{array} { r l } & { \underset { \lambda } { \operatorname* { m a x } } \mathbb { E } _ { \tau \sim x ( \cdot \vert q ) } \left[ \overset { \sum } { i \in N } \lambda _ { i } ( \tau ) r _ { i } ( \ell ( \tau ) ) + \beta \mathcal { H } ( \lambda ( \tau ) ) \right] } \\ & { ~ = \mathbb { E } _ { \tau \sim x ( \cdot \vert q ) } \left[ \overset { T _ { \tau } - 1 } { - \beta } \underset { t = 0 } { \operatorname* { l o g } } \frac { x \left( a _ { t } \mid s _ { t } \right) } { \pi _ { \mathrm { r e f } } \left( a _ { t } \mid s _ { t } \right) } \right] } \\ & { ~ = \mathbb { E } _ { \tau \sim x ( \cdot \vert q ) } \left[ G _ { \beta } ( \ell ( \tau ) ) - \beta \overset { T _ { \tau } - 1 } { t = 0 } \log \frac { x \left( a _ { t } \mid s _ { t } \right) } { \pi _ { \mathrm { r e f } } \left( a _ { t } \mid s _ { t } \right) } \right] . } \end{array}
$$

Th<sub>e</sub> inn<sub>e</sub>r m<sub>a</sub>ximiz<sub>e</sub>r i<sub>s</sub> $\lambda ^ { \beta } ,$ , while E<sub>q</sub>uation (3) shows that the outer m<sub>a</sub>ximiz<sub>e</sub>r i<sub>s</sub> $x ^ { \beta }$ <sub>.</sub> H<sub>e</sub>n<sub>ce</sub> $( x ^ { \beta } , \lambda ^ { \beta } )$ maximizes the stated entro<sub>py</sub>- <sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub>d <sub>we</sub>lf<sub>are.</sub>

W<sub>e now prove</sub> th<sub>e we</sub>lf<sub>are</sub> b<sub>oun</sub>d<sub>.</sub> F<sub>or every</sub> $\lambda \in \Delta ( N )$ <sub>,</sub> <sub>eac</sub>h term $- \lambda _ { i }$ <sup>l</sup>o<sub>g</sub> $\lambda _ { i }$ is nonne<sub>g</sub>ative<sub>,</sub> while

$$
\log | N | - { \mathcal { H } } ( \lambda ) = \mathbb { D } _ { \mathrm { K L } } { \big ( } \lambda \| U ( N ) { \big ) } \geq 0 .
$$

Therefore<sub>,</sub> in <sub>p</sub>artic<sub>u</sub>lar<sub>,</sub>

$$
0 \leq \mathcal { H } ( \lambda ^ { \beta } ( \ell ) ) \leq \log | { \cal N } | .
$$

Moreover<sub>,</sub> for ever<sub>y</sub> $i \in N$

$$
\begin{array} { l } { \displaystyle { G _ { \beta } ( \ell ) = \beta \log \sum _ { j \in \cal N } \exp \left( \frac { r _ { j } ( \ell ) } \beta \right) } } \\ { \displaystyle \ \geq \beta \log \exp \left( \frac { r _ { i } ( \ell ) } \beta \right) = r _ { i } ( \ell ) . } \end{array}
$$

Takin<sub>g</sub> the maximum over � <sub>g</sub>ives $G _ { \beta } ( \ell ) \geq \operatorname* { m a x } _ { i \in N } r _ { i } ( \ell )$ . Usin<sub>g</sub> E<sub>q</sub>uations (1) and (3), we obtain

$$
\begin{array} { r l } & { \quad \mathrm { S W } _ { p } ( x _ { i } ^ { 0 , \beta } ) ^ { l } \left\{ q _ { j } r \right\} } \\ & { = \sqrt { 1 - \kappa ( i \omega ) } \left[ \begin{array} { l } { C _ { i j } (( t ( i ) ) - \beta ) \displaystyle \sum _ { m = 1 } ^ { \tau - 1 } \log \frac { x \left( a _ { i } \left( s _ { i } \right) s _ { j } \right) } { \mathcal { K } _ { i j } \pi \left( a _ { i } \left( s _ { j } \right) s _ { j } \right) } } \\ { - \beta \displaystyle \sum _ { m = 1 } ^ { \tau } \exp ( a _ { i } \left( | a _ { i } ( s _ { i } ) | \right) ) } \end{array} \right] } \\ & { \quad - \beta \displaystyle \sum _ { c = \delta } \alpha _ { i } \left[ \gamma _ { i } ( a _ { i } ^ { 0 , \beta } ( \cdot ) ) \right] \frac { x } { \mathcal { K } _ { i j } \pi \left( a _ { i } \left( s _ { i } \right) - \beta \right) } \left[ \begin{array} { l } { C _ { i 1 } ^ { - 1 } } \\ { - 1 } \end{array} \right] } \\ & { \geq \mathrm { s u p } \displaystyle \sum _ { c = \delta } \log \frac { x } { \mathcal { K } _ { i j } \pi \left( a _ { i } \left( s _ { i } \right) - \beta \right) } \left[ \begin{array} { l } { 1 0 _ { c } \frac { x } { \mathcal { K } _ { i j } \pi \left( a _ { i } \left( s _ { i } \right) s _ { j } \right) } } \\ { - 1 } \end{array} \right] } \\ & { \quad - \beta \displaystyle \log ( \beta ) \Big [ 1 } \\ & { \geq \mathrm { s u p } \displaystyle \sum _ { c = \delta } \operatorname { t i m } _ { \tau } \left( \operatorname* { m a x } _ { i } ^ { 0 } r ( \cdot ) \right) - \beta \displaystyle \sum _ { m = 1 } ^ { \tau - 1 } \log \frac { x \left( a _ { i } \left( s _ { i } \right) s _ { j } \right) } { \mathcal { K } _ { i j } \pi \left( a _ { i } \left( s _ { i } \right) - \beta \right) } } \\ &  \quad - \beta \end{array}
$$

Th<sub>e</sub> l<sub>as</sub>t <sub>equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>ecause,</sub> <sub>con</sub>diti<sub>ona</sub>l <sub>on</sub> <sub>eac</sub>h t<sub>erm</sub>i<sub>na</sub>l state<sub>,</sub> the sim<sub>p</sub>lex first-best assi<sub>g</sub>ns the o<sub>pp</sub>ortunit<sub>y</sub> to an advertiser <sub>w</sub>ith m<sub>a</sub>xim<sub>a</sub>l <sub>va</sub>l<sub>ue.</sub> Sin<sub>ce</sub> $( x ^ { \beta } , \lambda ^ { \beta } )$ i<sub>s</sub> it<sub>se</sub>lf f<sub>eas</sub>ibl<sub>e</sub> f<sub>or</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>s</sub>i<sub>mp</sub>l<sub>ex-a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em,</sub>

$$
\operatorname { S W } _ { \beta } ( x ^ { \beta } , \lambda ^ { \beta } \mid q , r ) \leq \operatorname* { s u p } _ { \pi , \lambda } { \operatorname { S W } _ { \beta } ( \pi , \lambda \mid q , r ) } .
$$

Combinin<sub>g</sub> the last two dis<sub>p</sub>la<sub>y</sub>s <sub>g</sub>ives

$$
0 \leq \operatorname* { s u p } _ { \pi , \lambda } \mathrm { S W } _ { \beta } ( \pi , \lambda \mid q , r ) - \mathrm { S W } _ { \beta } ( x ^ { \beta } , \lambda ^ { \beta } \mid q , r ) \leq \beta \log \vert N \vert .
$$

Fi<sub>na</sub>ll<sub>y,</sub> l<sub>e</sub>t $\ell ^ { * }$ be a maximizin<sub>g</sub> terminal state satisf<sub>y</sub>in<sub>g</sub> $P _ { 0 } ( \ell ^ { * } \mid$ $q ) > 0 .$ . A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Lemma 1 to the terminal reward ma $\mathfrak { x } _ { i \in N } \mathfrak { r } _ { i } ( \ell )$

Token-Level Advertising

<sub>g</sub><sup>i</sup>ves

$$
\operatorname* { s u p } _ { \pi , \lambda } \mathrm { S W } _ { \beta } ( \pi , \lambda \mid q , r ) = \beta \log \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) \exp \left( { \frac { \operatorname* { m a x } _ { i \in N } r _ { i } ( \ell ) } { \beta } } \right) .
$$

It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t

$$
\begin{array} { r l } & { \underset { \ell \in L ( q ) } { \operatorname* { m a x } } \ \underset { i \in N } { \operatorname* { m a x } } r _ { i } ( \ell ) + \beta \log P _ { 0 } ( \ell ^ { * } \mid q ) } \\ & { \leq \underset { \pi , \lambda } { \operatorname* { s u p } } \mathrm { S W } _ { \beta } ( \pi , \lambda \mid q , r ) \leq \underset { \ell \in L ( q ) } { \operatorname* { m a x } } \ \underset { i \in N } { \operatorname* { m a x } } r _ { i } ( \ell ) . } \end{array}
$$

B<sub>o</sub>th b<sub>oun</sub>d<sub>s</sub> <sub>converge</sub> t<sub>o</sub> $\operatorname* { m a x } _ { \ell \in L ( q ) } \operatorname* { m a x } _ { i \in N } r _ { i } ( \ell )$ as $\beta \to 0 ^ { + }$ <sub>.</sub> T<sub>o-</sub> <sub>ge</sub>th<sub>er</sub> <sub>w</sub>ith th<sub>e</sub> <sub>we</sub>lf<sub>are-gap</sub> b<sub>oun</sub>d<sub>,</sub> th<sub>e</sub> <sub>squeeze</sub> th<sub>eorem</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\operatorname { S W } _ { \beta } ( x ^ { \beta } , \lambda ^ { \beta } \mid q , r ) \to \operatorname* { m a x } _ { \ell \in L ( q ) } \operatorname* { m a x } _ { i \in N } r _ { i } ( \ell ) .
$$

The welfare-<sub>g</sub>a<sub>p</sub> ine<sub>q</sub>ualit<sub>y</sub> is <sub>p</sub>ointwise in $q ,$ so takin<sub>g</sub> ex<sub>p</sub>ectation over $q \sim \mathcal { D }$ <sub>p</sub>reserves t<sup>h</sup>e same $\beta \log | N |$ b<sub>oun</sub>d<sub>.</sub> □

Proof of Theorem 3<sub>.</sub> Fix <sub>a</sub>n <sub>a</sub>d<sub>ve</sub>rti<sub>se</sub>r-<sub>que</sub>r<sub>y</sub> <sub>pa</sub>ir $( i , q )$ <sub>,</sub> <sub>an</sub>d l<sub>e</sub>t $\pi ^ { \dagger }$ <sup>b</sup>e any popu<sup>l</sup>ation minimizer o<sup>f</sup> t<sup>h</sup>e <sup>l</sup>oca<sup>l</sup>-a<sup>d</sup>vantage o<sup>b</sup>jective. D<sub>e</sub>fi<sub>ne</sub>

$$
\widehat { A } _ { i } ^ { \dagger } ( s , a ) : = \beta \log \frac { \pi _ { i } ^ { \dagger } ( a \mid s ) } { \pi _ { \mathrm { r e f } } ( a \mid s ) } .
$$

For a com<sub>p</sub>leted res<sub>p</sub>onse $y = \left( a _ { 0 } , \ldots , a _ { T _ { y } - 1 } \right)$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $s _ { 0 } = q , s _ { t + 1 } =$ $[ s _ { t } , a _ { t } ] ;$ <sub>,</sub> <sub>an</sub>d $s _ { T _ { y } } = \ell _ { y } ,$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> it<sub>s cumu</sub>l<sub>a</sub>ti<sub>ve score</sub>

$$
G _ { i } ^ { \dagger } ( q , y ) : = \sum _ { t = 0 } ^ { T _ { y } - 1 } \widehat { A } _ { i } ^ { \dagger } ( s _ { t } , a _ { t } ) .
$$

B<sub>y</sub> <sub>au</sub>t<sub>o</sub>r<sub>eg</sub>r<sub>ess</sub>i<sub>ve</sub> f<sub>ac</sub>t<sub>o</sub>riz<sub>a</sub>ti<sub>o</sub>n<sub>,</sub>

$$
\begin{array} { r l } & { G _ { i } ^ { \dagger } ( q , y ) = \displaystyle \sum _ { t = 0 } ^ { T _ { y } - 1 } \beta \log \frac { \pi _ { i } ^ { \dagger } ( a _ { t } \mid s _ { t } ) } { \pi _ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } } \\ & { \qquad = \beta \log \frac { \pi _ { i } ^ { \dagger } ( y \mid q ) } { \pi _ { \mathrm { r e f } } ( y \mid q ) } . } \end{array}
$$

F<sub>o</sub>r <sub>a</sub> <sub>co</sub>m<sub>pa</sub>ri<sub>so</sub>n <sub>pa</sub>ir $( y , y ^ { \prime } )$ <sub>,</sub> l<sub>e</sub>t

$$
\Delta _ { i } ^ { \dagger } ( q ; y , y ^ { \prime } ) : = G _ { i } ^ { \dagger } ( q , y ) - G _ { i } ^ { \dagger } ( q , y ^ { \prime } ) .
$$

The Bradle<sub>y</sub>-Terr<sub>y</sub> tar<sub>g</sub>et is

$$
\omega _ { i } ( q ; y , y ^ { \prime } ) = \sigma \left( r _ { i } ( \ell _ { y } ) - r _ { i } ( \ell _ { y ^ { \prime } } ) \right) .
$$

For a <sub>p</sub>redicted lo<sub>g</sub>it Δ, abbreviate $p : = \omega _ { i } ( q ; y , y ^ { \prime } )$ <sub>an</sub>d ${ \widehat { p } } _ { \Delta } : = \sigma ( \Delta )$ . Sin<sub>ce</sub> $\sigma ( - \Delta ) = 1 - \widehat { p } _ { \Delta } ,$ <sub>,</sub> the binar<sub>y</sub> cross-entro<sub>py</sub> satisfies

$$
\begin{array} { r l } & { - \ p \log \widehat { \mathcal { P } } _ { \Delta } - ( 1 - \gamma ) \log ( 1 - \widehat { \mathcal { P } } _ { \Delta } ) } \\ & { = \left[ - p \log \mathcal { P } - ( 1 - \mathcal { P } ) \log ( 1 - \mathcal { P } ) \right] } \\ & { \quad + \ p \log \displaystyle \frac { \mathcal { P } } { \widehat { \mathcal { P } } _ { \Delta } } + ( 1 - \mathcal { P } ) \log \displaystyle \frac { 1 - \mathcal { P } } { 1 - \widehat { \mathcal { P } } _ { \Delta } } } \\ & { = \mathcal { H } \big ( \mathrm { B e r n } ( \mathcal { P } ) \big ) + \mathbb { D } _ { \mathrm { K L } } \big ( \mathrm { B e r n } ( \mathcal { P } ) \big | \big | \mathrm { B e r n } ( \widehat { \mathcal { P } } _ { \Delta } ) \big ) . } \end{array}
$$

Th<sub>e m</sub>iddl<sub>e equa</sub>lit<sub>y a</sub>dd<sub>s an</sub>d <sub>su</sub>bt<sub>rac</sub>t<sub>s</sub> th<sub>e</sub> B<sub>ernou</sub>lli <sub>en</sub>t<sub>ropy</sub> t<sub>erms.</sub> $\mathrm { B y }$ nonne<sub>g</sub>ativit<sub>y</sub> of the KL diver<sub>g</sub>ence<sub>,</sub> the loss is minimized exactl<sub>y</sub> <sub>w</sub>h<sub>en</sub> $\widehat { \pmb { p } } _ { \Delta } = \pmb { p }$

B<sub>y</sub> Lemma 1<sub>,</sub> the o<sub>p</sub>timal <sub>p</sub>olic<sub>y</sub> satisfies

$$
\beta \log { \frac { \pi _ { i } ^ { * } ( a \mid s ) } { \pi _ { \mathrm { r e f } } ( a \mid s ) } } = V _ { i } ^ { * } ( [ s , a ] ) - V _ { i } ^ { * } ( s ) .
$$

Telesco<sub>p</sub>in<sub>g</sub> alon<sub>g</sub> a com<sub>p</sub>leted res<sub>p</sub>onse <sub>g</sub>ives

$$
\beta \log \frac { \pi _ { i } ^ { * } ( y \mid q ) } { \pi _ { \mathrm { r e f } } ( y \mid q ) } = r _ { i } ( \ell _ { y } ) - V _ { i } ^ { * } ( q ) .
$$

H<sub>e</sub>n<sub>ce</sub> $\pi _ { i } ^ { * }$ induces the tar<sub>g</sub>et lo<sub>g</sub>it $r _ { i } ( \ell _ { y } ) - r _ { i } ( \ell _ { y ^ { \prime } } )$ <sub>.</sub> B<sub>ecause</sub> $\pi _ { i } ^ { * }$ i<sub>s</sub> f<sub>ea-</sub> sible<sub>,</sub> the <sub>p</sub>ointwise lower bound is attainable<sub>,</sub> so ever<sub>y</sub> <sub>p</sub>o<sub>p</sub>ulation minimiz<sub>e</sub>r <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\sigma \left( \Delta _ { i } ^ { \dagger } ( q ; y , y ^ { \prime } ) \right) = \sigma \left( r _ { i } ( \ell _ { y } ) - r _ { i } ( \ell _ { y ^ { \prime } } ) \right)
$$

<sup>f</sup>or every pair o<sup>f</sup> comp<sup>l</sup>ete<sup>d</sup> responses. Since � is injective,

$$
G _ { i } ^ { \dagger } ( q , y ) - G _ { i } ^ { \dagger } ( q , y ^ { \prime } ) = r _ { i } ( \ell _ { y } ) - r _ { i } ( \ell _ { y ^ { \prime } } ) .
$$

Th<sub>ere</sub>f<sub>ore</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a cons</sub>t<sub>an</sub>t $c _ { i } ( \boldsymbol { q } )$ <sub>suc</sub>h th<sub>a</sub>t

$$
G _ { i } ^ { \dagger } ( q , y ) = r _ { i } ( \ell _ { y } ) + c _ { i } ( q ) .\tag{4}
$$

E<sub>q</sub>uation (4) identifies cumulative scores u<sub>p</sub> to the <sub>q</sub>uer<sub>y</sub>-s<sub>p</sub>ecific constant $c _ { i } ( \boldsymbol { q } )$ . We next determine this constant from token-<sub>p</sub>olic<sub>y</sub> n<sub>o</sub>rm<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n<sub>.</sub>

N<sub>o</sub>rm<sub>a</sub>liz<sub>a</sub>ti<sub>o</sub>n <sub>o</sub>f $\pi _ { i } ^ { \dagger }$ i<sub>mp</sub>li<sub>es, a</sub>t <sub>every reac</sub>h<sub>a</sub>bl<sub>e non-</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>pre</sub>fi<sub>x �,</sub>

$$
\sum _ { a \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( \frac { \widehat { A } _ { i } ^ { \dagger } ( s , a ) } { \beta } \right) = \sum _ { a \in \mathcal { V } } \pi _ { i } ^ { \dagger } ( a \mid s ) = 1 .\tag{5}
$$

B<sub>ecause</sub> t<sub>o</sub>k<sub>en</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>are</sub> <sub>pre</sub>fi<sub>xes,</sub> <sub>every</sub> <sub>reac</sub>h<sub>a</sub>bl<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> h<sub>as</sub> <sub>a</sub> <sub>un</sub>i<sub>que</sub> <sub>pa</sub>th f<sub>rom</sub> th<sub>e</sub> <sub>roo</sub>t<sub>.</sub> F<sub>or</sub> $s _ { t } = \left[ q , a _ { < t } \right]$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\begin{array} { c } { { \displaystyle \widetilde { V } _ { i } ( s _ { t } ) : = - c _ { i } ( q ) + \sum _ { k = 0 } ^ { t - 1 } \widehat { A } _ { i } ^ { \dagger } ( s _ { k } , a _ { k } ) , } } \\ { { \widetilde { V } _ { i } ( [ s , a ] ) : = \widetilde { V } _ { i } ( s ) + \widehat { A } _ { i } ^ { \dagger } ( s , a ) . } } \end{array}\tag{6}
$$

At <sub>a</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $\ell _ { y } ,$ E<sub>q</sub>uation (4) <sub>g</sub>ives

$$
\widetilde { V } _ { i } ( \ell _ { y } ) = - c _ { i } ( q ) + G _ { i } ^ { \dagger } ( q , y ) = r _ { i } ( \ell _ { y } ) .
$$

For ever<sub>y</sub> non-terminal <sub>p</sub>refix �, E<sub>q</sub>uations (5) and (6) im<sub>p</sub>l<sub>y</sub>

$$
\begin{array} { l } { { \displaystyle \beta \log \sum _ { a \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( \frac { \widetilde { V } _ { i } ( { \scriptstyle \lfloor s , a \rfloor } ) } { \beta } \right) } } \\ { { \displaystyle = \widetilde { V } _ { i } ( s ) + \beta \log \sum _ { a \in \mathcal { V } } \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( \frac { \widehat { A } _ { i } ^ { \dagger } ( s , a ) } { \beta } \right) = \widetilde { V } _ { i } ( s ) } . } \end{array}
$$

Th<sub>us</sub> $\widetilde { V _ { i } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e so</sub>ft B<sub>e</sub>ll<sub>man sys</sub>t<sub>em</sub> i<sub>n</sub> L<sub>emma</sub> 1<sub>.</sub> It<sub>s</sub> fi<sub>n</sub>it<sub>e-</sub> h<sub>or</sub>i<sub>zon so</sub>l<sub>u</sub>ti<sub>on</sub> i<sub>s un</sub>i<sub>que</sub> b<sub>y</sub> b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l b<sub>oun</sub>d<sub>ary,</sub> <sub>so</sub>

$$
\widetilde { V } _ { i } ( s ) = V _ { i } ^ { * } ( s )
$$

<sub>a</sub>t <sub>every</sub> <sub>reac</sub>h<sub>a</sub>bl<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e.</sub> I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub>

$$
- c _ { i } ( q ) = \widetilde { V } _ { i } ( q ) = V _ { i } ^ { * } ( q ) .\tag{7}
$$

W<sub>e can now</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>scores.</sub> F<sub>or a</sub> fi<sub>xe</sub>d <sub>pa</sub>i<sub>r</sub> $( s , a )$ <sub>,</sub> write a continuation trajectory <sup>f</sup>rom t<sup>h</sup>e c<sup>h</sup>i<sup>ld</sup> state as $\tau = ( u _ { 0 } , b _ { 0 } , \ldots . u _ { T _ { \tau } } )$ <sub>w</sub>h<sub>ere</sub> $u _ { 0 } = [ s , a ]$ <sub>.</sub> Th<sub>e op</sub>ti<sub>ma</sub>l <sub>so</sub>ft <sub>a</sub>d<sub>van</sub>t<sub>age</sub> $A _ { i } ^ { * } ( s ,$ �)ofcommittin<sub>g</sub> t<sub>o � a</sub>t <sub>�</sub> i<sub>s common</sub>l<sub>y</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub>

$$
\operatorname* { s u p } _ { \pi } \mathbb { E } _ { \tau \sim \pi ( \cdot | u _ { 0 } ) } \left[ r _ { i } ( u _ { T _ { \tau } } ) - \beta \sum _ { k = 0 } ^ { T _ { \tau } - 1 } \log \frac { \pi ( b _ { k } \mid u _ { k } ) } { \pi _ { \mathrm { r e f } } ( b _ { k } \mid u _ { k } ) } \right] - V _ { i } ^ { * } ( s ) .
$$

B<sub>ecause</sub> th<sub>e</sub> tr<sub>a</sub>n<sub>s</sub>iti<sub>o</sub>n i<sub>s</sub> d<sub>e</sub>t<sub>e</sub>rmini<sub>s</sub>ti<sub>c a</sub>nd th<sub>e</sub>r<sub>e a</sub>r<sub>e</sub> n<sub>o</sub> int<sub>e</sub>rm<sub>e</sub>di<sub>a</sub>t<sub>e</sub> <sub>rewar</sub>d<sub>s,</sub> th<sub>e supremum</sub> i<sub>s</sub> $V _ { i } ^ { * } ( [ s , a ] )$ <sub>.</sub> Th<sub>us</sub> $A _ { i } ^ { * } ( s , a ) = V _ { i } ^ { * } ( [ s , a ] ) -$ $V _ { i } ^ { * } ( s )$ , and E<sub>q</sub>uation (6) <sub>g</sub>ives

$$
\widehat { A } _ { i } ^ { \dagger } ( s , a ) = \widetilde { V } _ { i } ( [ s , a ] ) - \widetilde { V } _ { i } ( s ) = V _ { i } ^ { * } ( [ s , a ] ) - V _ { i } ^ { * } ( s ) = A _ { i } ^ { * } ( s , a ) .\tag{8}
$$

Usin<sub>g</sub> the <sub>p</sub>olic<sub>y</sub> formula in Lemma 1<sub>,</sub> the definition of $\widehat { A } _ { i } ^ { \dagger }$ <sub>, an</sub>d E<sub>q</sub>uation (8),

$$
\begin{array} { l } { \pi _ { i } ^ { \dagger } ( a \mid s ) = \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( \displaystyle { \frac { \widehat { A } _ { i } ^ { \dagger } ( s , a ) } { \beta } } \right) } \\ { = \pi _ { \mathrm { r e f } } ( a \mid s ) \exp \left( \displaystyle { \frac { A _ { i } ^ { * } ( s , a ) } { \beta } } \right) } \\ { = \pi _ { i } ^ { * } ( a \mid s ) . } \end{array}
$$

Th<sub>us</sub> $\pi _ { i } ^ { \dagger } = \pi _ { i } ^ { * }$ <sub>.</sub> Sin<sub>ce</sub> $( i , q )$ was arbitrar<sub>y,</sub> the <sub>p</sub>o<sub>p</sub>ulation minimizer is uni<sub>q</sub>ue in induced <sub>p</sub>olic<sub>y</sub> s<sub>p</sub>ace.

It remains to identif<sub>y</sub> the root value used to initialize the led<sub>g</sub>er. Combinin<sub>g</sub> E<sub>q</sub>uations (4) and (7) <sub>g</sub>ives, for ever<sub>y</sub> com<sub>p</sub>leted re-<sup>s</sup>p<sup>onse</sup> �,

$$
r _ { i } ( \ell _ { y } ) - \sum _ { t = 0 } ^ { T _ { y } - 1 } \widehat { A } _ { i } ^ { \dagger } ( s _ { t } , a _ { t } ) = r _ { i } ( \ell _ { y } ) - G _ { i } ^ { \dagger } ( q , y ) = - c _ { i } ( q ) = V _ { i } ^ { * } ( q ) .
$$

The residual is <sub>p</sub>ointwise e<sub>q</sub>ual to $V _ { i } ^ { * } ( q )$ <sub>.</sub> T<sub>a</sub>kin<sub>g</sub> it<sub>s co</sub>nditi<sub>o</sub>n<sub>a</sub>l ex<sub>p</sub>ectation therefore <sub>g</sub>ives

$$
\begin{array} { r } { v _ { i } ^ { \dagger } ( q ) = \mathbb { E } _ { y \sim \mathcal { D } ( \cdot | i , q ) } \left[ r _ { i } ( \ell _ { y } ) - \displaystyle \sum _ { t = 0 } ^ { T _ { y } - 1 } \widehat { A } _ { i } ^ { \dagger } ( s _ { t } , a _ { t } ) \right] = V _ { i } ^ { \ast } ( q ) . } \end{array}
$$

Thus the <sub>p</sub>o<sub>p</sub>ulation root value is recovered on the trainin<sub>g</sub> su<sub>pp</sub>ort.

With the root value identified<sub>,</sub> initialize the servin<sub>g</sub>-time led<sub>g</sub>er <sup>b</sup><sub>y</sub>

$$
\widehat V _ { i } ( q ) : = v _ { i } ^ { \dagger } ( q ) = V _ { i } ^ { * } ( q )
$$

and u<sub>p</sub>date it recursivel<sub>y</sub> accordin<sub>g</sub> to

$$
\widehat V _ { i } ( [ s , a ] ) : = \widehat V _ { i } ( s ) + \widehat A _ { i } ^ { \dagger } ( s , a ) .
$$

If $\widehat { V } _ { i } ( s ) = V _ { i } ^ { * } ( s )$ <sub>,</sub> th<sub>en</sub>

$$
\begin{array} { r l } & { \widehat { V } _ { i } ( \left[ s , a \right] ) = \widehat { V } _ { i } ( s ) + \widehat { A } _ { i } ^ { \dagger } ( s , a ) } \\ & { \phantom { \widehat { V } _ { i } ( \left[ s , a \right] ) = \widehat { V } _ { i } ^ { * } ( } s \overline { { \upsilon } } + V _ { i } ^ { * } ( \left[ s , a \right] ) - V _ { i } ^ { * } ( s ) = V _ { i } ^ { * } ( \left[ s , a \right] ) . } \end{array}
$$

Induction over <sub>p</sub>refix len<sub>g</sub>th <sub>g</sub>ives $\widehat { V } _ { i } ( s ) = V _ { i } ^ { * } ( s )$ <sub>a</sub>t <sub>every</sub> <sub>reac</sub>h<sub>a</sub>bl<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e, so</sub> th<sub>e cons</sub>t<sub>ruc</sub>t<sub>e</sub>d <sub>c</sub>hild<sub>-va</sub>l<sub>ue vec</sub>t<sub>or</sub> i<sub>s</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>so</sub>ft<sub>-va</sub>l<sub>ue</sub> re<sub>p</sub>ort re<sub>qu</sub>ired b<sub>y</sub> LAMA. □

Proof of Proposition 1. Fix a <sub>qu</sub>er<sub>y</sub> <sub>�</sub> and candidate ad<sub>v</sub>ertiser set N. Under exact re<sub>p</sub>ortin<sub>g</sub>, s<sub>p</sub>ecialize the desirabilit<sub>y</sub> nota ti<sub>on o</sub>f L<sub>emma</sub> 3 t<sub>o</sub> th<sub>e</sub> t<sub>ru</sub>thf<sub>u</sub>l <sub>va</sub>l<sub>ue pro</sub>fil<sub>e:</sub>

$$
z _ { i } ^ { * } ( q ) : = \exp \left( \frac { V _ { i } ^ { * } ( q ) } { \beta } \right) , \ : \overline { { { z } } } ( q ; V ^ { * } ) = \sum _ { j \in N } z _ { j } ^ { * } ( q ) .
$$

Th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\rho _ { i } ( q ) = \frac { z _ { i } ^ { * } ( q ) } { \overline { { z } } ( q ; V ^ { * } ) } .
$$

F<sub>or</sub> <sub>a</sub> <sub>rea</sub>li<sub>ze</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s T ,$ l<sub>e</sub>t

$$
P _ { i } ^ { \mathrm { f r a c } } ( s _ { T } ) : = \rho _ { i } ( s _ { T } ) r _ { i } ( s _ { T } ) - \beta \log \overline { { z } } ( q ; V ^ { * } ) + \beta \log \bigl ( 1 + \overline { { z } } ( q ; V ^ { * } ) - z _ { i } ^ { * } ( q ) \bigr )
$$

d<sub>eno</sub>t<sub>e a</sub>d<sub>ver</sub>ti<sub>ser</sub> �’<sub>s</sub> t<sub>e</sub>l<sub>escope</sub>d f<sub>rac</sub>ti<sub>ona</sub>l <sub>paymen</sub>t b<sub>e</sub>f<sub>ore</sub> th<sub>e w</sub>i<sub>nner-</sub> pro<sup>b</sup>a<sup>b</sup>i<sup>l</sup>ity a<sup>d</sup>justment in A<sup>l</sup>gorit<sup>h</sup>m 1. T<sup>h</sup>e winner-sett<sup>l</sup>e<sup>d</sup> payment i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub>

$$
P _ { i } ( s _ { T } , I ^ { * } ) = 1 \{ I ^ { * } = i \} \frac { P _ { i } ^ { \mathrm { f r a c } } ( s _ { T } ) } { \rho _ { i } ( s _ { T } ) } , I ^ { * } \sim \rho ( s _ { T } ) .
$$

We first <sub>p</sub>rove outcome individual rationalit<sub>y</sub>. Because terminal val<sub>u</sub>es are nonne<sub>g</sub>ative<sub>,</sub> Lemma 1 <sub>g</sub>ives

$$
\begin{array} { l } { { \displaystyle z _ { i } ^ { * } ( q ) = \exp \left( \frac { V _ { i } ^ { * } ( q ) } { \beta } \right) } } \\ { { \displaystyle \ = \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) \exp \left( \frac { r _ { i } ( \ell ) } { \beta } \right) \geq 1 . } } \end{array}
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>, $\overline { { z } } ( q ; V ^ { * } ) \geq 1 + \overline { { z } } ( q ; V ^ { * } ) - z _ { i } ^ { * } ( q )$

Su<sub>pp</sub>ose advertiser � is the sam<sub>p</sub>led winner at terminal state $s _ { T }$ B<sub>y</sub> exactness of the terminal re<sub>p</sub>ort<sub>,</sub> $V _ { i } ^ { * } ( s _ { T } ) = r _ { i } ( s _ { T } )$ <sub>, so</sub> it<sub>s rea</sub>li<sub>ze</sub>d <sub>u</sub>tilit<sub>y</sub> i<sub>s</sub>

$$
r _ { i } ( s _ { T } ) - P _ { i } ( s _ { T } , i ) = \frac { \beta } { \rho _ { i } ( s _ { T } ) } \log \left( \frac { \overline { { z } } ( q ; V ^ { * } ) } { 1 + \overline { { z } } ( q ; V ^ { * } ) - z _ { i } ^ { * } ( q ) } \right) \geq 0 ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e prece</sub>di<sub>ng</sub> d<sub>es</sub>i<sub>ra</sub>bilit<sub>y</sub> b<sub>oun</sub>d<sub>.</sub> Th<sub>us</sub> th<sub>e w</sub>i<sub>nner</sub>’<sub>s paymen</sub>t d<sub>oes no</sub>t <sub>excee</sub>d it<sub>s rea</sub>li<sub>ze</sub>d <sub>ga</sub>i<sub>n.</sub> If <sub>a</sub>d<sub>-</sub> vertiser � is not selected<sub>,</sub> it receives no advertisin<sub>g</sub> o<sub>pp</sub>ortunit<sub>y</sub> and <sub>p</sub>a<sub>y</sub>s zero<sub>,</sub> so its realized utilit<sub>y</sub> is also zero. Therefore outcome IR h<sub>o</sub>ld<sub>s</sub> <sub>po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se</sub> f<sub>or</sub> <sub>every</sub> <sub>rea</sub>li<sub>ze</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub>d <sub>every</sub> <sub>samp</sub>l<sub>e</sub>d <sub>w</sub>inn<sub>e</sub>r<sub>.</sub>

W<sub>e</sub> <sub>nex</sub>t <sub>prove</sub> <sub>expec</sub>t<sub>e</sub>d <sub>wea</sub>k b<sub>u</sub>d<sub>ge</sub>t b<sub>a</sub>l<sub>ance.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> <sub>a</sub> <sub>rea</sub>li<sub>ze</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $s _ { T }$ <sub>,</sub> takin<sub>g</sub> ex<sub>p</sub>ectation onl<sub>y</sub> over the sam<sub>p</sub>led <sub>w</sub>inn<sub>e</sub>r <sub>g</sub>i<sub>ves</sub>

$$
\mathbb { E } _ { I ^ { * } \sim \rho ( s _ { T } ) } \left[ P _ { i } ( s _ { T } , I ^ { * } ) \mid s _ { T } \right] = \rho _ { i } ( s _ { T } ) \frac { P _ { i } ^ { \mathrm { f r a c } } ( s _ { T } ) } { \rho _ { i } ( s _ { T } ) } = P _ { i } ^ { \mathrm { f r a c } } ( s _ { T } ) .
$$

Th<sub>us</sub> it <sub>su</sub>fi<sub>ces</sub> t<sub>o s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e expec</sub>t<sub>e</sub>d f<sub>rac</sub>ti<sub>ona</sub>l <sub>paymen</sub>t<sub>s are</sub> nonne<sub>g</sub>at<sup>i</sup>ve.

B<sub>y</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $P _ { i } ^ { \mathrm { f r a c } }$

$$
\begin{array} { r l r } {  { \frac { 1 } { \beta } \mathbb { E } _ { s _ { T } \sim x ^ { \beta } } [ P _ { i } ^ { \mathrm { f r a c } } ( s _ { T } ) ] = \frac { 1 } { \beta } \mathbb { E } _ { s _ { T } \sim x ^ { \beta } } [ \rho _ { i } ( s _ { T } ) r _ { i } ( s _ { T } ) ] - \log \overline { { z } } ( q ; V ^ { * } ) } } \\ & { } & { + \log \bigl ( 1 + \overline { { z } } ( q ; V ^ { * } ) - z _ { i } ^ { * } ( q ) \bigr ) . } \end{array}
$$

Th<sub>e</sub> l<sub>as</sub>t t<sub>wo</sub> t<sub>erms are</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c, so</sub> th<sub>e on</sub>l<sub>y ran</sub>d<sub>om</sub> t<sub>erm</sub> th<sub>a</sub>t <sub>mus</sub>t b<sub>e</sub> <sub>con</sub>t<sub>ro</sub>ll<sub>e</sub>d i<sub>s</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>a</sub>ll<sub>oca</sub>ti<sub>on-we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>va</sub>l<sub>ue.</sub> B<sub>y</sub> Lemma 4<sub>,</sub>

$$
\begin{array} { l } { { \displaystyle { \mathbb E } _ { s _ { T } \sim x ^ { \beta } } \left[ \rho _ { i } ( s _ { T } ) r _ { i } ( s _ { T } ) \right] } } \\ { { \displaystyle \quad = \frac 1 { \overline { { z } } ( q ; V ^ { * } ) } \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) \exp \left( \frac { r _ { i } ( \ell ) } \beta \right) r _ { i } ( \ell ) } . } \end{array}
$$

T<sub>o</sub> <sub>norma</sub>li<sub>ze</sub> thi<sub>s</sub> <sub>sum,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub>

$$
\nu _ { i } ( \ell ) : = \frac { P _ { 0 } ( \ell \mid q ) \exp ( r _ { i } ( \ell ) / \beta ) } { z _ { i } ^ { * } ( q ) } .
$$

This is a <sub>p</sub>robabilit<sub>y</sub> distribution because the leaf re<sub>p</sub>resentation above <sub>g</sub>ives $\begin{array} { r } { z _ { i } ^ { * } ( q ) = \sum _ { \ell \in L ( q ) } P _ { 0 } ( \ell \mid q ) } \end{array}$ ex<sub>p</sub>(�<sub>�</sub> (ℓ)/�). The ex<sub>p</sub>ected <sub>a</sub>ll<sub>oca</sub>ti<sub>on-we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>va</sub>l<sub>ue can now</sub> b<sub>e wr</sub>itt<sub>en as</sub>

$$
\mathbb { E } _ { s _ { T } \sim x ^ { \beta } } \left[ \rho _ { i } ( s _ { T } ) r _ { i } ( s _ { T } ) \right] = \frac { z _ { i } ^ { * } ( q ) } { \overline { { z } } ( q ; V ^ { * } ) } \mathbb { E } _ { \ell \sim \nu _ { i } } [ r _ { i } ( \ell ) ] .
$$

It <sub>rema</sub>i<sub>ns</sub> t<sub>o</sub> l<sub>ower-</sub>b<sub>oun</sub>d th<sub>e expec</sub>t<sub>a</sub>ti<sub>on un</sub>d<sub>er</sub> $\nu _ { i } .$ . For ever<sub>y</sub> ℓ <sub>w</sub>ith $\nu _ { i } ( \ell ) > 0$ <sub>,</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f $\nu _ { i }$ im<sub>p</sub>lies

$$
\log \frac { \nu _ { i } ( \ell ) } { P _ { 0 } ( \ell \mid q ) } = \frac { r _ { i } ( \ell ) } { \beta } - \log z _ { i } ^ { * } ( q ) .
$$

Takin<sub>g</sub> ex<sub>p</sub>ectation under $\nu _ { i }$ an<sup>d</sup> rearran<sub>g</sub><sup>i</sup>n<sub>g</sub> <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r } { \mathbb { E } _ { \ell \sim \nu _ { i } } \left[ r _ { i } ( \ell ) \right] - \beta \mathbb { D } _ { \mathrm { K L } } \big ( \nu _ { i } \| P _ { 0 } ( \cdot \lfloor q ) \big ) = \beta \log z _ { i } ^ { * } ( q ) = V _ { i } ^ { * } ( q ) . } \end{array}
$$

Sin<sub>ce</sub> th<sub>e</sub> KL di<sub>ve</sub>r<sub>ge</sub>n<sub>ce</sub> i<sub>s</sub> n<sub>o</sub>nn<sub>ega</sub>ti<sub>ve,</sub>

$$
\begin{array} { r } { \mathbb { E } _ { \ell \sim \nu _ { i } } \left[ r _ { i } ( \ell ) \right] \geq V _ { i } ^ { \ast } ( q ) = \beta \log z _ { i } ^ { \ast } ( q ) . } \end{array}
$$

Combinin<sub>g</sub> the last ine<sub>q</sub>ualit<sub>y</sub> with the re<sub>p</sub>resentation ofthe allocation-<sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>va</sub>l<sub>ue</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\mathbb { E } _ { s _ { T } \sim x ^ { \beta } } \left[ \rho _ { i } ( s _ { T } ) r _ { i } ( s _ { T } ) \right] \geq \beta \frac { z _ { i } ^ { * } ( q ) } { \overline { { z } } ( q ; V ^ { * } ) } \log z _ { i } ^ { * } ( q ) .
$$

Substitutin<sub>g</sub> this bound into the <sub>p</sub>a<sub>y</sub>ment decom<sub>p</sub>osition <sub>g</sub>ives

$$
\begin{array} { r l r } & { \frac { 1 } { \beta } \mathbb { E } [ P _ { i } ^ { \mathrm { f r a c } } ( s _ { T } ) ] \ge \frac { z _ { i } ^ { * } ( q ) } { \overline { { z } } ( q ; V ^ { * } ) } \log z _ { i } ^ { * } ( q ) - \log \overline { { z } } ( q ; V ^ { * } ) } & \\ & { } & { + \log \bigl ( 1 + \overline { { z } } ( q ; V ^ { * } ) - z _ { i } ^ { * } ( q ) \bigr ) . } \end{array}
$$

It remains to show that the ri<sub>g</sub>ht-hand side is nonne<sub>g</sub>ative. Since $0 < z _ { i } ^ { * } ( q ) / \overline { { z } } ( q ; V ^ { * } ) \leq 1$ <sub>,</sub> the wei<sub>g</sub>hted arithmic-<sub>g</sub>eometric mean ine<sub>q</sub>ualit<sub>y</sub> <sub>g</sub>ives

$$
\begin{array} { r l r } {  { \frac { 1 + \overline { { z } } ( q ; V ^ { * } ) - z _ { i } ^ { * } ( q ) } { \overline { { z } } ( q ; V ^ { * } ) } = ( 1 - \frac { z _ { i } ^ { * } ( q ) } { \overline { { z } } ( q ; V ^ { * } ) } ) \cdot 1 + \frac { z _ { i } ^ { * } ( q ) } { \overline { { z } } ( q ; V ^ { * } ) } \cdot \frac { 1 } { z _ { i } ^ { * } ( q ) } } } \\ & { } & { \geq 1 ^ { 1 ^ { 1 - z _ { i } ^ { * } ( q ) / \overline { { z } } ( q ; V ^ { * } ) } } ( \frac { 1 } { z _ { i } ^ { * } ( q ) } ) ^ { z _ { i } ^ { * } ( q ) / \overline { { z } } ( q ; V ^ { * } ) } } \\ & { } & { = z _ { i } ^ { * } ( q ) ^ { - z _ { i } ^ { * } ( q ) / \overline { { z } } ( q ; V ^ { * } ) } . } \end{array}
$$

Takin<sub>g</sub> lo<sub>g</sub>arithms<sub>,</sub>

$$
\begin{array} { r } { \log \bigl ( 1 + \overline { { z } } ( q ; V ^ { * } ) - z _ { i } ^ { * } ( q ) \bigr ) - \log \overline { { z } } ( q ; V ^ { * } ) \geq - \frac { z _ { i } ^ { * } ( q ) } { \overline { { z } } ( q ; V ^ { * } ) } \log z _ { i } ^ { * } ( q ) . } \end{array}
$$

Th<sub>ere</sub>f<sub>ore,</sub> $\mathbb { E } [ P _ { i } ^ { \mathrm { f r a c } } ( s _ { T } ) ] \geq 0$ <sup>f</sup>or ever<sub>y</sub> $i \in N$

Finall<sub>y,</sub> takin<sub>g</sub> ex<sub>p</sub>ectations in the conditional <sub>p</sub>a<sub>y</sub>ment identit<sub>y</sub> above and summin<sub>g</sub> over advertisers <sub>g</sub>ives

$$
\begin{array} { r l } & { \mathbb { E } _ { s _ { T } , I ^ { * } } \left[ \displaystyle \sum _ { i \in { \cal N } } { P _ { i } ( s _ { T } , I ^ { * } ) } \right] = \displaystyle \sum _ { i \in { \cal N } } \mathbb { E } _ { s _ { T } } { [ P _ { i } ^ { \mathrm { f r a c } } ( s _ { T } ) ] } } \\ & { \quad \quad \quad \quad \quad \quad \ge 0 . } \end{array}
$$

Th<sub>e expec</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s over</sub> b<sub>o</sub>th t<sub>o</sub>k<sub>en-genera</sub>ti<sub>on ran</sub>d<sub>omness an</sub>d th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>w</sub>i<sub>nner</sub> d<sub>raw, con</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e query an</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e</sub> advertiser set. Hence the winner-settled im<sub>p</sub>lementation is weakl<sub>y</sub> b<sub>u</sub>d<sub>ge</sub>t b<sub>a</sub>l<sub>ance</sub>d i<sub>n</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on.</sub> □

## D Additional Experiment Results

This a<sub>pp</sub>endix <sub>p</sub>rovides su<sub>pp</sub>ortin<sub>g</sub> ex<sub>p</sub>erimental details and di-<sub>agnos</sub>ti<sub>cs.</sub> T<sub>a</sub>bl<sub>e</sub> 2 <sub>summar</sub>i<sub>zes</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>ver</sub>ti<sub>ca</sub>l<sub>s an</sub>d th<sub>e</sub>i<sub>r</sub> advertisers<sub>,</sub> and Tables 3–5 <sub>p</sub>resent re<sub>p</sub>resentative <sub>g</sub>enerated re-<sub>sponses</sub> f<sub>or</sub> <sub>eac</sub>h <sub>ver</sub>ti<sub>ca</sub>l<sub>.</sub> T<sub>a</sub>bl<sub>es</sub> 6 <sub>an</sub>d 7 ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub> h<sub>ow</sub> th<sub>e</sub> <sub>genera</sub>t<sub>e</sub>d res<sub>p</sub>onse chan<sub>g</sub>es with bids for com<sub>p</sub>lementar<sub>y</sub> and com<sub>p</sub>etin<sub>g</sub> advertiser <sub>p</sub>airs<sub>,</sub> res<sub>p</sub>ectivel<sub>y</sub>. Table 8 com<sub>p</sub>ares the end-to-end latenc<sub>y</sub> <sub>o</sub>f LAMA <sub>w</sub>ith <sub>re</sub>f<sub>erence-answer genera</sub>ti<sub>on.</sub> T<sub>a</sub>bl<sub>e</sub> 9 <sub>an</sub>d Fi<sub>gure</sub> 4 <sub>repor</sub>t th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>an</sub>d <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>errors</sub> <sub>o</sub>f th<sub>e</sub> l<sub>earne</sub>d <sub>s</sub>h<sub>are</sub>d re<sub>p</sub>ort model<sub>,</sub> while Fi<sub>g</sub>ure 5 em<sub>p</sub>iricall<sub>y</sub> evaluates its monotonicit<sub>y</sub> <sub>un</sub>d<sub>er</sub> bid <sub>per</sub>t<sub>ur</sub>b<sub>a</sub>ti<sub>ons.</sub>

Hanbing Liu, Bowei Zhang, Changyuan Yu, Yinyu Ye, and Qi Qi
<table><tr><td>Vertical</td><td># Queries</td><td>Description Fitness and workout-</td><td>Advertisers</td><td>Type</td><td>Qualities</td></tr><tr><td rowspan="3">Workout</td><td rowspan="3">291</td><td rowspan="3">intent search queries, covering supplements, gym routines, class dis- covery, and home exer- cise equipment.</td><td>Bowflex SelectTech 552Adjustable dumbbells</td><td></td><td>Replace a full rack of weights with adjustable dumbbells for home strength training.</td></tr><tr><td>C4 Sport</td><td>Pre-workout supplement</td><td>Prepare for demanding gym sessions with a fa- miliar pre-workout op- tion.</td></tr><tr><td>ClassPass</td><td>Gym and studio marketplace</td><td>Book gyms, barre, Pi- lates, boxing, and cy- cling classes with flexi- ble choices.</td></tr><tr><td rowspan="3">Vacation</td><td rowspan="3">456</td><td rowspan="3">queries, including destinations, packages, flights, hotels, cruises, resorts, and itinerary planning. Automotive search queries for vehicle research, used-car</td><td>Expedia</td><td>Travel booking platform</td><td>Compare flights, hotels, cars, and vacation pack- ages for flexible trip planning. Search travel deals</td></tr><tr><td>Priceline</td><td>Travel deal booking platform</td><td>across hotels, flights, rental cars, and vaca- tion packages. Compare destinations,</td></tr><tr><td>Tripadvisor</td><td>Travel review and planning platform</td><td>resorts, attractions, and traveler reviews while planning a trip. Check vehicle value</td></tr><tr><td rowspan="3">Car 492</td><td rowspan="3"></td><td rowspan="3">shopping, pricing/value checks, EV/hybrid comparison, and dealer- local intents.</td><td>Kelley Blue Book</td><td>Vehicle valuation and pricing research service</td><td>ranges, pricing guid- ance, and market information when comparing cars. Compare vehicle re-</td></tr><tr><td>Edmunds</td><td>Vehicle research and review platform</td><td>views, specs, pricing guidance, and own- ership factors when researching cars.</td></tr><tr><td>CARFAX</td><td>Vehicle history service</td><td>Review vehicle history details when evaluating used cars or comparing specific vehicles.</td></tr></table>

Table 2: Vertical overview and advertiser metadata used in the experiments.

Token-Level Advertising
<table><tr><td>Field</td><td>Content</td></tr><tr><td>Vertical</td><td>Workout</td></tr><tr><td>Query</td><td>best bicep exercises for mass</td></tr><tr><td>Reference</td><td>The best bicep exercises for mass typically include compound and isolation movements that target the biceps effectively. Common options are barbell curls, dumbbell curls, chin-ups, hammer curls, and concentration curls. These exercises vary in intensity and muscle engagement, allowing for progressive overload, which is essential for building mass. Key factors to consider include proper form, controlled movements, and sufficient resistance to challenge the muscles. It&#x27;s also important to incorporate a variety of exercises to stimulate different parts of the biceps and avoid plateaus. Training frequency, rest periods, and nutrition play crucial roles in muscle growth. For optimal results, pair these exercises with a well-rounded upper body routine and adequate protein intake. Always ensure that exercises are performed with correct technique to prevent injury, and consider consulting a fitness professional for personalized guidance.</td></tr><tr><td>LAMA</td><td>The best bicep exercises for mass focus on progressive overload and proper form to effectively build size and strength. Common effective moves include barbell curls, dumbbell curls, cable chin-ups or pull-downs, hammer curls, and concentration curls. These exercises target the biceps brachii from different angles, promoting muscle hypertrophy. Key factors to prioritize are using heavy enough weights for 6-12 reps, maintaining a full range of motion, and ensuring rest and nutrition support growth.For those with limited space, @Bowflex SelectTech 552@ can be an excellent choice as they offer adjustable weights that make performing a variety of bicep-focused lifts more convenient at home.Incorporating a mix of free weights and machine-based exercises can lead to more balanced development. Also, supersetting bicep workouts with opposing tricep movements may enhance muscle growth. Always warm up adequately to reduce injury risk and maximize performance.</td></tr><tr><td>Winner</td><td>Bowflex</td></tr></table>

Table 3: Case Study 1: Workout vertical. In the generated content, the winning advertiser is shown in bold, and the highlighted sentence is the span allocated to the hyperlink.

<table><tr><td rowspan=1 colspan=7>Field      Content</td></tr><tr><td rowspan=1 colspan=7>Vertical    Vacation</td></tr><tr><td rowspan=1 colspan=7>Query      best beach vacations in the world</td></tr><tr><td rowspan=7 colspan=7>Reference  The best beach vacations in the world offer a mix of pristine sands, clear waters, and unique cultural experiences, appealing to travelersseeking relaxation or adventure. Popular destinations include the Maldives for luxury overwater bungalows, Bali for vibrant culture andnatural beauty, and the Caribbean for family-friendly resorts and vibrant nightlife. Key factors to consider include travel timing, with peakseasons often running from late spring to early fall, and budget, which can vary greatly depending on whether you opt for all-inclusiveresorts, boutique hotels, or local guesthouses. Location preferences also play a role, with some travelers favoring secluded spots and othersprioritizing proximity to activities or nightlife. Flexibility in travel dates can lead to better deals and availability, while last-minute bookingsmay require more adaptable options. Practical considerations include researching visa requirements, health advisories, and weather patterns,and ensuring that travel insurance covers potential disruptions. Whether planning a romantic getaway, family trip, or solo adventure,comparing package deals, rental options, and local experiences can help tailor the perfect beach vacation to individual needs and constraints</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=7>LAMA     The best beach vacations in the world offer a mix of stunning scenery, clear waters, and diverse activities tailored to different preferences@Expedia@ highlights top options like the Maldives for luxury overwater bungalows, Bali for cultural richness and vibrant nightlife, andthe Caribbean for family-friendly resorts and cruise access. White-sand beaches in Greece, such as those in Santorini or Mykonos, appeal</td></tr><tr><td rowspan=2 colspan=7>to couples seeking romance, while Mexico&#x27;s Riviera Maya offers adventure and world-class resorts. Key factors include seasonality, withpeak times often from June to August, affecting budget and availability. Travelers should consider visa requirements, health advisories,and local conditions. For flexibility, all-inclusive resorts or cruise packages may suit families or solo travelers, while last-minute optionsabound through online travel agencies. Beach vacationers should also plan for sun protection, water activities, and transportation to remote</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7>islands.@Tripadvisor@ can help compare destinations, read traveler reviews, and find the best accommodations and experiences that</td></tr><tr><td rowspan=1 colspan=7>match personal preferences and travel styles.Always verify current travel advisories and check with providers for the latest updates onservices and restrictions.</td></tr><tr><td rowspan=1 colspan=7>Winner    Tripadvisor</td></tr></table>

Table 4: Case Study 2: Vacation vertical. In the generated content, the winning advertiser is shown in bold, and the highlighted sentence is the span allocated to the hyperlink.

Token-Level Advertising
<table><tr><td>Field</td><td>Content</td></tr><tr><td>Vertical</td><td>Car</td></tr><tr><td>Query</td><td>cars for sale near me</td></tr><tr><td>Reference</td><td>Find cars for sale near you by checking local dealerships or online listings that highlight nearby inventory. Many dealers offer in-person visits, virtual tours, and flexible scheduling to help you explore options conveniently. When looking for cars for sale near me, consider visiting multiple dealers to compare available models, pricing, and financing options. Local dealers often carry a range of new and used vehicles, including popular brands like Toyota, Honda, and Ford, and can provide details about vehicle history, warranty coverage, and service records. It&#x27;s also a good idea to confirm current availability and pricing directly with the dealership, as inventory and offers can change frequently. Some dealers may offer test drives, maintenance packages, or special promotions to help you make an informed decision. Whether you&#x27;re looking for a new car, used car, hybrid, or electric vehicle, checking with local dealers can provide a personalized shopping experience tailored to your needs and preferences. Be sure to ask about any current incentives or financing deals that may be available to you.</td></tr><tr><td>LAMA</td><td>If you&#x27;re looking for cars for sale near you, checking with local dealers is the best way to find current inventory. @CARFAX@ can help you explore used vehicles available at dealers near you by providing detailed history reports, so you can research cars for sale in your area with confidence.Local dealers typically offer in-person visits, test drives, and financing options, making it easier to compare models and secure a deal. To locate nearby dealers with suitable options, you can use your device&#x27;s map service or the search function on automotive websites. Be sure to call ahead to confirm availability and set up a visit, as popular models may sell quickly. If you&#x27;re interested in a specific make or model, dealers often have multiple units on-site or can order one for you. Used cars for sale near you usually come with a range of options, including different years, trims, and mileage, so you can find a fit for your needs and budget. @Kelley Blue Book@ is another resource that can help you understand fair pricing for used vehicles at dealers near you. By combining these tools with direct contact to local dealers, you can streamline your search for the right car in your area.</td></tr><tr><td>Winner</td><td>CARFAX</td></tr></table>

Table 5: Case Study 3: Car vertical. In the generated content, the winning advertiser is shown in bold, and the highlighted sentence is the span allocated to the hyperlink.

<table><tr><td rowspan=1 colspan=5>Offset   λ Content</td></tr><tr><td rowspan=3 colspan=4>-0.2 0.007</td><td rowspan=1 colspan=1>Used cars are a popular choice for buyers seeking reliable transportation at a lower cost, offering a range of models, years, and conditions.</td></tr><tr><td rowspan=1 colspan=1>@CARFAX@ is a trusted resource that provides detailed insights into a used car&#x27;s history, including accident reports, service records, and</td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=1 colspan=1>ownership changes, which can be especially helpful when you&#x27;re considering a specific used car.The key factors to consider include the</td></tr><tr><td rowspan=6 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>or Toyota Corolla, their reputation for reliability and fuel efficiency makes them top contenders. Buyers should also pay attention to the</td></tr><tr><td rowspan=1 colspan=1>seller type—whether it&#x27;s a private seller or a certified used dealer—as this can impact the car&#x27;s condition and included services. A thorough</td></tr><tr><td rowspan=1 colspan=1>inspection by an independent mechanic is recommended before finalizing a purchase. Additionally, checking for any outstanding liens or</td></tr><tr><td rowspan=1 colspan=1>hidden damages ensures a smoother transaction. Researching recent model-year versions can also help in comparing current market values</td></tr><tr><td rowspan=1 colspan=1>and features. Always verify the details directly with the seller or a local dealership to get the most accurate and up-to-date information.</td></tr><tr><td rowspan=1 colspan=4>0 0.074 If yo</td><td rowspan=1 colspan=1>u&#x27;re looking for used cars, start by evaluating models based on their mileage, condition, and pricing history, with a focus on popular</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>options like the Honda Civic, Toyota Corolla, and Hyundai Elantra, which typically offer strong resale value and long-term reliability. Used</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>cars from 2015 to 2020 are especially common and often provide a good balance of modern features and affordability. @CARFAX@ is a</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>valuable resource that can help you review the history of used cars, including accident reports, service records, and ownership details, which</td></tr><tr><td rowspan=4 colspan=4></td><td rowspan=1 colspan=1>can inform your decision-making.When shopping, consider the type of seller—dealerships often provide certified pre-owned vehicles with</td></tr><tr><td rowspan=1 colspan=1>warranties, while private sellers may offer lower prices but less protection. Be sure to inspect the car thoroughly, ideally with a professional</td></tr><tr><td rowspan=1 colspan=1>mechanic, and ask for a detailed vehicle history report. The best deals can be found by comparing listings across trusted platforms and</td></tr><tr><td rowspan=1 colspan=1>understanding the local market to identify fair pricing and hidden value.</td></tr><tr><td rowspan=1 colspan=4>+0.2 0.926 If you</td><td rowspan=1 colspan=1>&#x27;re looking for used cars, start by evaluating models based on their mileage, condition, and pricing history, with popular options</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>including the Honda Civic, Toyota Corolla, and Hyundai Elantra, which are known for their dependability and value in the pre-owned</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>market. Used cars typically range in price from a few thousand to over $30,000, depending on the year, trim, and features, and it&#x27;s essential</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>to review their complete vehicle history for any past accidents or maintenance records.@Kelley Blue Book@ is a helpful resource that</td></tr><tr><td rowspan=4 colspan=4></td><td rowspan=1 colspan=1>provides insights into used cars for buyers, offering tools to check fair market values, set realistic asking prices, and understand current</td></tr><tr><td rowspan=5 colspan=4></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1>trends.When considering a used car, always inspect the vehicle thoroughly or have a qualified mechanic perform a pre-purchase inspection,</td></tr><tr><td rowspan=1 colspan=1>especially for models with higher mileage. @CARFAX@ can also provide a detailed look at a used car&#x27;s history, including title status and</td></tr><tr><td rowspan=1 colspan=1>service records, so you can make a more informed decision before finalizing a purchase.</td></tr><tr><td rowspan=3 colspan=5>Table 6: Car vertical, query &quot;used cars&quot;, with participating advertisers Kelley Blue Book (KBB) and CARFAX. KBB primarilycontributes used-car pricing and market-value guidance, while CARFAX contributes vehicle-history and title/service-recordevidence; the pair is mainly complementary for used-car shopping, with limited overlap as automotive information resources.KBB- and CARFAX-generated tokens are underlined in their respective colors. In the generated content, the winning advertiseris shown in bold, and the highlighted sentence is the span allocated to the hyperlink.</td></tr><tr><td rowspan=1 colspan=1>resources</td></tr><tr><td rowspan=1 colspan=3>KBB- and</td><td rowspan=1 colspan=1>advertiser</td></tr></table>

<table><tr><td rowspan=1 colspan=2>Offset   λ Content</td></tr><tr><td rowspan=8 colspan=1>-0.2 0.019</td><td rowspan=1 colspan=1>If you&#x27;re looking for used cars, start by evaluating models based on their mileage, condition, and pricing history, with popular options</td></tr><tr><td rowspan=1 colspan=1>including the Honda Civic, Toyota Corolla, and Hyundai Elantra, which are known for their reliability and affordability. Used cars from 2015</td></tr><tr><td rowspan=1 colspan=1>to 2020 typically offer a good balance of modern features and lower prices, and their value depends largely on maintenance records, accident</td></tr><tr><td rowspan=1 colspan=1>history, and overall exterior and interior condition.@Edmunds@ can help you explore detailed listings and pricing guidance for used cars,</td></tr><tr><td rowspan=1 colspan=1>providing insights into fair market value and common features across different years and trims.Be sure to check the vehicle history report</td></tr><tr><td rowspan=1 colspan=1>for any red flags and consider scheduling a pre-purchase inspection by a certified mechanic to ensure the car you&#x27;re buying is in solid shape.</td></tr><tr><td rowspan=1 colspan=1>When shopping for used cars, compare sellers like private individuals, dealerships, and certified pre-owned listings, as each offers different</td></tr><tr><td rowspan=1 colspan=1>levels of warranty and service support, and always test drive the vehicle to confirm it meets your expectations for performance and comfort.</td></tr><tr><td rowspan=8 colspan=1>0 0.546</td><td rowspan=1 colspan=1>If you&#x27;re looking for used cars, start by evaluating models based on their mileage, condition, and pricing history, with popular options</td></tr><tr><td rowspan=1 colspan=1>including the Honda Civic, Toyota Corolla, and Hyundai Elantra, which are known for their reliability and affordability. Used cars from 2015</td></tr><tr><td rowspan=1 colspan=1>to 2020 typically offer a good balance of modern features and lower prices, and their value depends on factors like maintenance records,</td></tr><tr><td rowspan=1 colspan=1>accident history, and service history.@Kelley Blue Book@ is a trusted resource that can help you understand the right price range for</td></tr><tr><td rowspan=1 colspan=1>used cars based on their year, model, trim, mileage, and location. @Edmunds@ can also assist in exploring detailed insights about used</td></tr><tr><td rowspan=1 colspan=1>cars, including common mechanical issues, repair costs, and owner reviews, so you can make a more informed decision. Be sure to get a</td></tr><tr><td rowspan=1 colspan=1>professional inspection before finalizing a purchase, especially for used cars with higher mileage, to ensure there are no hidden mechanical</td></tr><tr><td rowspan=1 colspan=1>problems that could increase your long-term costs.</td></tr><tr><td rowspan=7 colspan=1>+0.2 0.974</td><td rowspan=1 colspan=1>If you&#x27;re looking for used cars, start by evaluating models based on their mileage, condition, and pricing history, with popular options</td></tr><tr><td rowspan=1 colspan=1>including the Honda Civic, Toyota Corolla, and Hyundai Elantra, which are known for their dependability and value in the pre-owned</td></tr><tr><td rowspan=1 colspan=1>market. Used cars typically range in price from a few thousand to over $30,000, depending on the year, trim, and features, and it&#x27;s essential</td></tr><tr><td rowspan=1 colspan=1>to review a comprehensive vehicle history report to ensure there are no hidden issues.@Kelley Blue Book@ is a trusted resource that</td></tr><tr><td rowspan=1 colspan=1>provides detailed pricing guidance for used cars, offering insights into fair market values and what to expect when you&#x27;re ready to negotiate.</td></tr><tr><td rowspan=1 colspan=1>Be sure to inspect the car thoroughly or have a qualified mechanic do so, and consider the seller type—whether a private individual or a</td></tr><tr><td rowspan=1 colspan=1>certified dealer—as this can impact the overall reliability and warranty options available with the used cars you&#x27;re considering.</td></tr></table>

Table 7: Car vertical, query “used cars”, with participating advertisers Kelley Blue Book (KBB) and Edmunds. Both advertisers can answer used-car research and pricing-comparison intent, so the pair is more competitive than KBB–CARFAX; they are partially complementary because KBB emphasizes valuation while Edmunds emphasizes reviews, model research, and shopping guidance. KBB- and Edmunds-generated tokens are underlined in their respective colors. In the generated content, the winning advertiser is shown in bold, and the highlighted sentence is the span allocated to the hyperlink.

Table 8: Latency and seconds-per-token ratios (LAMA/Reference) across verticals. These ratios are close to the theoretical prediction (|N | + 1)
<table><tr><td>Vertical</td><td>Latency Mean</td><td>p50</td><td>p90</td><td>Sec/Token</td></tr><tr><td>Workout</td><td>3.87×</td><td>3.90×</td><td>4.31×</td><td>3.51×</td></tr><tr><td>Vacation</td><td>3.53×</td><td>3.47×</td><td>3.90×</td><td>3.53×</td></tr><tr><td>Car</td><td>3.94×</td><td>3.97×</td><td>4.56×</td><td>3.57×</td></tr></table>

Table 9: Impression-value prediction and calibration performance across verticals. In our experiments, impression values lie in [0, 1] and are defined as click-through rates multiplied by a unit cost per click. The current experimental setup (Qwen3-14B + LoRA fine-tuning) already yields reasonably good recovery accuracy.

<table><tr><td>Vertical</td><td>MAE</td><td>Rel. MAE</td><td>ECE</td></tr><tr><td>Workout</td><td>0.052</td><td>7.1%</td><td>0.0149</td></tr><tr><td>Vacation</td><td>0.065</td><td>8.5%</td><td>0.0300</td></tr><tr><td>Car</td><td>0.056</td><td>7.4%</td><td>0.0375</td></tr><tr><td>Macro Avg.</td><td>0.058</td><td>7.7%</td><td>0.0275</td></tr></table>

![](images/e5799c7245b2e31a7a3532e71b687547b67563695ffdae50f4690784eb3b92d9.jpg)

Figure 4: Quantile-binned calibration curves for the value model �ˆ across the three verticals. Each panel corresponds to one vertical, and each point represents one of 50 equal-frequency bins sorted by predicted impression value. The x-axis reports the mean predicted impression value within the bin, while the y-axis reports the mean ground-truth impression value for the same examples.  
![](images/645c66142bfb8262e9124c65675318171600b20b9d382a383fa49e82374548c1.jpg)  
Figure 5: Bid-ofset sweep for the LAMA token auction across the three verticals. For each vertical, we select one target advertiser and add a constant ofset to its truthful value $v ( q )$ while leaving the other advertisers’ bids unchanged. The target advertisers are CARFAX for car, ClassPass for workout, and Tripadvisor for vacation. The sweep ranges from −0.3 to +0.3, while the advantage model settings remain fixed to the vertical-specific configuration. The four panels report the mean target token share, final allocation probability, expected payment, and true expected utility. Across all three verticals, increasing the bid ofset monotonically increases token share, allocation, and payment. In contrast, true utility peaks at the truthful bid for all three verticals, then declines under overbidding as higher payments ofset the additional allocation. This pattern provides qualitative evidence that the mechanism rewards truthful bidding while still allowing bids to control token-level allocation.