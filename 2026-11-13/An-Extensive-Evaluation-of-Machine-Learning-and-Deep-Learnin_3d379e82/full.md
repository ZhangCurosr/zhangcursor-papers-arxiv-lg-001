# An Extensive Evaluation of Machine Learning and Deep Learning Approaches for Sentiment Analysis: Insights from Amazon Customer Reviews

Muhammed Ihsan Corak <sup>1</sup>, Cuiyuan Wang <sup>2</sup> and Xiaowen Zhang <sup>3</sup>\* <sup>1</sup> Computer Science Dept., CUNY College of Staten Island m.ihsan.corak@gmail.com

<sup>2</sup> Computer Science Dept., CUNY LaGuardia Community College cuiyuanwang18@gmail.com

<sup>3</sup> Computer Science Dept., CUNY College of Staten Island xiaowen.zhang@csi.cuny.edu

## Abstract

The rapid growth of user-generated text on online platforms creates opportunities for understanding customer opinion and supporting data-driven decisions. This study compares five traditional machine learning classifiers (Random Forest, Logistic Regression, Bernoulli Naive Bayes, Linear Support Vector Classifier, and Multinomial Naive Bayes) and six deep learning architectures (LSTM, Bi-LSTM, GRU, RNN, Attention, and FastText) for three-class sentiment classification of 568,454 Amazon Fine Food Reviews. Machine learning models used Bag-of-Words (BoW) and TF-IDF features limited to 5,000 terms; deep learning models used learned embeddings over a 20,000-word vocabulary. Models were evaluated on the original imbalanced data and on a random-oversampled balanced set of 1,070,469 reviews, using accuracy, macro and per-class precision, recall, and F1-score, and training time. On the balanced set, Random Forest with BoW achieved 99% accuracy and the Attention model achieved 98.7%. Linear SVC reached 89% accuracy in 1 min 15 s, and Logistic Regression reached 90% in 3 min 37 s. On the imbalanced set, the Attention model achieved 97.8% accuracy, and FastText achieved 95.3% in 3 min 26 s, while the RNN reached 91.9% with a neutral-class F1-score of 0.08. Deep learning models were more stable across class distributions, whereas the best classical model depended on balancing. We also report a user-facing application that applies the trained Random Forest and Attention models to new review text. Because the labels are derived from star ratings, the results measure agreement with rating-derived sentiment classes rather than annotated sentiment; the dataset is a single source, and the findings should be read with that limitation in mind.

Keywords: Deep Learning; Machine Learning; Sentiment Analysis; E-commerce; Social Media; Amazon; Data Analytics; Decision Making; Natural Language Processing

## 1 Introduction

In today’s data-driven world, online reviews have become an essential factor influencing consumer purchasing decisions and guiding business strategies. With the increasing dependence on digital platforms, customers frequently share their experiences and opinions online, providing organizations with valuable insights into customer satisfaction, product quality, and emerging market trends. Among these platforms, Amazon stands out as one of the largest e-commerce ecosystems, hosting millions of user-generated reviews that form a rich and diverse source of unstructured textual data. Analyzing such large-scale customer feedback systematically can enable businesses to improve service quality, optimize product oferings, and strengthen customer engagement.

Sentiment analysis, a subfield of Natural Language Processing (NLP), focuses on identifying and classifying opinions expressed in text as positive, negative, or neutral. It has become a powerful analytical tool for extracting emotional tone from textual data and converting subjective information into actionable business intelligence. Traditional machine learning models including Random Forest Classifier (RFC), Logistic Regression Classifier (LRC), Bernoulli Naive Bayes (BNB), Multinomial Naive Bayes (MNB) and Linear Support Vector Classifier (L-SVC) have long been used to perform sentiment classification. These approaches rely on manually crafted features and text representation methods like Bag of Words (BoW) and Term Frequency–Inverse Document Frequency (TF-IDF). While these models are eficient and interpretable, their ability to capture deep semantic meaning and contextual relationships in language remains limited.

In recent years, deep learning models have demonstrated superior performance in sentiment analysis due to their capability to learn hierarchical representations directly from text data. Architectures such as Long Short-Term Memory (LSTM), Bidirectional Long Short-Term Memory (Bi-LSTM), Gated Recurrent Units (GRU), Recurrent Neural Networks (RNN), and Attention-based mechanisms can efectively capture long-range dependencies and contextual nuances. Word embeddings such as FastText further enhance model understanding by encoding semantic relationships between words. These developments have significantly advanced the field, making deep learning models particularly efective for complex text classification tasks. However, they often come with higher computational costs and longer training times, which may afect scalability and real-time applicability.

This study presents a comprehensive analysis of sentiment classification on Amazon customer reviews [1] by integrating both traditional machine learning and deep learning approaches. Using a large dataset containing over half a million customer reviews, the research systematically evaluates multiple models trained with BoW and TF-IDF features as well as deep learning models leveraging word embeddings. A key component of this work is the examination of model performance across both balanced and imbalanced datasets to assess the efect of data distribution on classification accuracy and fairness. To enhance computational eficiency, experiments were conducted in a GPU T4 x2 environment, utilizing two NVIDIA T4 GPUs for parallel training and accelerated model optimization.

Beyond comparative performance analysis, a practical sentiment analysis software system was developed to demonstrate real-time classification capabilities and user interaction. This integration of research and application underscores the importance of combining methodological rigor with practical usability. The findings from this study ofer a holistic perspective on the relative strengths, limitations, and trade-ofs between machine learning and deep learning models. By providing insights into model selection, data balancing strategies, and computational eficiency, this research contributes a unified framework for large-scale sentiment analysis and ofers guidance for both academic research and industry applications.

The remainder of this paper is organized as follows. Section 2 reviews related work on sentiment analysis of Amazon reviews and identifies the gap this study addresses. Section 3 describes the dataset, preprocessing, and data balancing procedures. Section 4 presents the machine learning models and their results. Section 5 presents the deep learning models and their results. Section 6 compares the two families under balanced and imbalanced conditions. Section 7 reports the best-performing models and the application prototype, and Section 8 discusses the findings and limitations. Section 9 concludes.

## 2 Literature Review

The studies reviewed above share three recurring characteristics. First, most evaluate a small number of models, often one classical and one neural architecture, so that the relative standing of the wider model families is unclear. Second, class imbalance is noted as important but is rarely tested systematically: the balancing strategy, its efect on each model family, and its validity (for example, whether oversampling leaks information into evaluation) are seldom examined. Third, the labels used are typically star ratings, and the mismatch between rating and expressed sentiment, particularly for three-star reviews, is acknowledged but not quantified in terms of its efect on model comparison.

This study addresses these gaps in three ways. It evaluates eleven models across two feature representations on the same large corpus. It reports results on both imbalanced data and a balanced set constructed by random oversampling, and it reports how each family responds to that change. It also documents the labeling scheme and the class composition explicitly, so that the results can be interpreted in terms of what the labels measure.

Sentiment analysis has evolved from basic text classification to advanced machine learning and deep learning techniques enabling more accurate and eficient interpretation of usergenerated textual data. With the rise of digitalization, large-scale datasets such as Amazon reviews ofer valuable opportunities to benchmark and compare diferent sentiment classification methodologies, helping address challenges like dataset bias through balancing techniques. Studies across platforms-from e-commerce to social media-highlight innovations such as ensemble methods and multi-polarity classification, all contributing to the development of more efective models and reinforcing sentiment analysis as a crucial tool for extracting meaningful insights and supporting informed business decision-making.

The study [2] compares machine learning algorithms for sentiment classification on Amazon reviews, using a Kaggle dataset with 4 million reviews. LSTM networks outperformed other models, achieving 90% accuracy and 0.96 Area Under Curve (AUC), with even higher accuracy (92%) for furniture reviews. Unlike the reviewed study, which excludes neutral sentiments and uses TF-IDF for feature extraction, this project includes neutral sentiments and employs both TF-IDF and BoW methods. The findings highlight LSTM’s efectiveness across product categories and its potential for expansion to include neutral sentiment classification.

Another study [3] investigates sentiment analysis on Amazon mobile phone reviews, highlighting the challenge of ambiguous 3-star ratings. Using models like LRC, Naive Bayesian (NB), RFC, Bi-LSTM with Global Vectors (GloVe) embeddings, and Bidirectional Encoder Representations from Transformers (BERT), the research identifies BERT as the most effective model, achieving 94% accuracy for multiclass and 98% for binary classification. Although sharing similarities with this project in sentiment classification, it difers by focusing on mobile phone reviews, emphasizing 3-star ratings, and employing GloVe embeddings and BERT. This contrasts with this project’s analysis of food reviews and its use of uses VADER, TF-IDF, and BoW for feature extraction.

The reviewed research [4] analyzes Amazon product reviews using GRU, RNNs, and Support Vector Machine (SVM) on 28,000 reviews from 60 products, achieving 81.82% accuracy by integrating review embeddings with product data. It highlights the importance of machine learning in understanding customer feedback and dataset balancing, with suggestions for future work in expanding product categories. While both projects focus on sentiment classification using Amazon reviews, this project uses over 500,000 reviews with TF-IDF and BoW, whereas the reviewed study VADER relies on star ratings meanwhile, this project applies sentiment scoring for a more nuanced analysis.

Previous studies have taken diverse approaches to sentiment analysis. One significant contribution was made by a study that proposed a Pearson Correlation Coeficient-based Harris Hawks Optimization (PCCHHO) algorithm combined with RNN and LSTM models [5]. This approach emphasized feature selection using the Pearson correlation coeficient for initial dimensionality reduction, followed by Harris Hawks Optimization for non-redundant feature selection. The use of RNN-LSTM aimed to classify sentiments with high precision, achieving 95.8% accuracy on Amazon product reviews. The study highlighted the efectiveness of feature extraction and optimization techniques in improving sentiment classification but did not focus on user-centric interface.

Another study employed a CNN model for classifying text reviews into positive or negative categories [6]. The research compared CNN with several word embedding models, such as Word2Vec (Word to Vector), GloVe, and FastText, finding that the inclusion of stop words improved accuracy by 2%. The study emphasized the benefits of training word embeddings to learn better features in less computation time. Although the CNN-based approach demonstrated strong performance, it doesn’t includes a user interface and did not explore multi-class sentiment classification.

The study utilized supervised machine learning techniques to polarize Amazon reviews into categories using bag-of-words and TF-IDF feature extraction methods combined with chi-square analysis [7]. This research focused on active learning strategies to label large datasets and achieved satisfactory results through manual and automatic learning processes. Despite its emphasis on feature extraction and supervised learning, the study did not delve into advanced deep learning architectures or the benefits of contextual embeddings for sentiment analysis.

The work by [8] provides a comprehensive survey of sentiment analysis techniques, covering both traditional machine learning and deep learning approaches. Traditional machine learning techniques, such as NB, SVM, and LRC, have been widely used for sentiment classification but require extensive feature engineering and struggle with capturing contextual dependencies. Deep learning models, particularly RNNs, LSTM networks, and CNNs, have demonstrated superior performance in handling sequential text data and learning complex linguistic features. Additionally, the study discusses the impact of pre-trained language models such as BERT and GPT, which have significantly improved sentiment analysis by leveraging transfer learning. The limitations of deep learning models, including computational cost and data dependency, are also highlighted. This survey provides an important foundation for understanding the advancements in sentiment analysis and emphasizes the need for comparative studies across diferent model architectures.

A related study by [9] presents a comparative analysis of sentiment classification models applied to Amazon customer reviews, similar to the focus of this research. The authors evaluate traditional machine learning models, including LRC, RFC, and NB, against DL architectures such as CNNs and RNNs. Their findings suggest that deep learning models tend to outperform traditional approaches, particularly in capturing nuanced sentiment variations. However, the study also notes that machine learning models remain competitive, ofering lower computational costs and better interpretability. Additionally, the authors emphasize the importance of dataset balancing techniques, as class imbalance in sentiment datasets can significantly impact model performance. This study aligns with our research objectives by exploring the trade-ofs between machine learning and deep learning models in sentiment classification.

Further insights are provided by [10], which reviews recent advancements in machine learning and deep learning for sentiment analysis. The study categorizes sentiment analysis models based on their structural complexity and evaluates their efectiveness in various applications, including social media monitoring and customer feedback analysis. The authors discuss emerging challenges in the field, such as the need for better context understanding, handling sarcasm, and improving model interpretability. Their findings underscore the potential of attention mechanisms and transformer-based architectures in enhancing sentiment classification accuracy. Additionally, they highlight the growing trend of hybrid models that combine ML and DL techniques to leverage the strengths of both approaches.

Overall, the existing body of research demonstrates a clear progression from traditional machine learning techniques toward more advanced deep learning and transformer-based architectures, driven by the increasing complexity and scale of sentiment-rich datasets. While earlier studies highlight the efectiveness of models such as NB, SVM, LRC, and even CNNs for binary or domain-specific tasks, more recent work consistently shows that architectures like LSTM, Bi-LSTM, GRU, and BERT excel at capturing contextual and sequential nuances in text. At the same time, challenges such as class imbalance, ambiguous ratings, and the need for eficient feature extraction continue to shape methodological choices across studies. Despite difering datasets, feature engineering strategies, and model architectures, the reviewed literature collectively emphasizes the importance of comparative evaluations to understand trade-ofs between accuracy, eficiency, and interpretability. These insights underscore the need for comprehensive, large-scale analyses-such as the one undertaken in this study-to systematically assess model performance across machine learning and deep learning methods while incorporating balanced datasets and multi-class sentiment perspectives.

## 3 Materials and Methods

## 3.1 Data Ingestion

The dataset utilized in this study consists of Amazon Fine Food Reviews collected over a thirteen-year period, spanning from October 1999 to October 2012. This dataset serves as a comprehensive resource for sentiment analysis research, as it encompasses both textual review content and associated metadata, including product identifiers, user information, and numerical ratings. In total, the dataset contains 568,454 individual reviews, reflecting diverse customer opinions and experiences across a broad range of products.

Each review record includes attributes such as the product ID, user ID, profile name, helpfulness score, review score, review summary, and detailed text review. The distribution of reviews across star ratings reveals a notable imbalance: 363,122 reviews correspond to 5- star ratings, 80,655 to 4-star, 42,640 to 3-star, 29,769 to 2-star, and 52,268 to 1-star ratings, as illustrated in Figure 1. Figure 2 presents the oversampled balanced dataset. The data balancing techniques will be discussed in the subsequent sections. This skew toward higher ratings is typical of e-commerce feedback datasets, highlighting the need for balanced analysis techniques.

Additionally, the dataset comprises 256,059 unique users and 74,258 distinct products, indicating a high degree of user and product diversity. While the primary focus is on fine food products, the dataset also includes reviews from other Amazon categories, contributing to a more generalized representation of customer sentiment. The inclusion of heterogeneous product categories enhances the dataset’s applicability for large-scale sentiment modeling and domain-independent text analysis.

![](images/8887156f7f6b4e86865fc2d4c6ff113ac48583d7ce194cdddc2bf521de326a6d.jpg)  
Figure 1: Count Reviews by Stars

![](images/338c7a3fb877a490f1d13c3261888bd11d69fc31472ab6b40a53b4d453daa1c0.jpg)  
Figure 2: Oversampled Dataset

## 3.2 Sentiment and Review Count Over Time

Figure 3 shows, the dataset spans over a decade, capturing reviews from October 1999 to October 2012. Analyzing the review counts over this period reveals relatively similar levels from the early 1999 until 2007. However, a noticeable upward trend begins between 2007 and 2009, followed by a significant increase from 2009 to 2012. This shift may reflect changes in customer behavior during this time, potentially correlated with the rapid growth of e-commerce platforms and online food delivery services. These developments likely influenced consumer engagement and review activity, as evidenced by the significant increase in sentiment score counts in the dataset. The findings highlight the interaction between technological developments and customer feedback trends and ofer insights into changing consumer behavior in this transformative period.

![](images/c28756ce0338dedf00d3c23ca9d24dd8d8ceff8bad44f743ae393386a467353c.jpg)

Figure 3: Review Count Over Time
<table><tr><td rowspan=1 colspan=2>Positive</td><td rowspan=1 colspan=2>Negative</td><td rowspan=1 colspan=2>Neutral</td></tr><tr><td rowspan=1 colspan=1>Word</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Word</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Word</td><td rowspan=1 colspan=1>Count</td></tr><tr><td rowspan=1 colspan=1>good</td><td rowspan=1 colspan=1>200,638</td><td rowspan=1 colspan=1>never</td><td rowspan=1 colspan=1>39,275</td><td rowspan=1 colspan=1>fine</td><td rowspan=1 colspan=1>16,110</td></tr><tr><td rowspan=1 colspan=1>great</td><td rowspan=1 colspan=1>167,175</td><td rowspan=1 colspan=1>bad</td><td rowspan=1 colspan=1>32,186</td><td rowspan=1 colspan=1>grocery store</td><td rowspan=1 colspan=1>12,679</td></tr><tr><td rowspan=1 colspan=1>love</td><td rowspan=1 colspan=1>127,520</td><td rowspan=1 colspan=1>disappointed</td><td rowspan=1 colspan=1>16,624</td><td rowspan=1 colspan=1>first time</td><td rowspan=1 colspan=1>7,795</td></tr><tr><td rowspan=1 colspan=1>even</td><td rowspan=1 colspan=1>75,000</td><td rowspan=1 colspan=1>taste like</td><td rowspan=1 colspan=1>11,651</td><td rowspan=1 colspan=1>normal</td><td rowspan=1 colspan=1>7,265</td></tr><tr><td rowspan=1 colspan=1>well</td><td rowspan=1 colspan=1>73,408</td><td rowspan=1 colspan=1>awful</td><td rowspan=1 colspan=1>5,109</td><td rowspan=1 colspan=1>even though</td><td rowspan=1 colspan=1>6,755</td></tr><tr><td rowspan=1 colspan=1>delicious</td><td rowspan=1 colspan=1>41,471</td><td rowspan=1 colspan=1>worse</td><td rowspan=1 colspan=1>3,654</td><td rowspan=1 colspan=1>okay</td><td rowspan=1 colspan=1>5,880</td></tr><tr><td rowspan=1 colspan=1>perfect</td><td rowspan=1 colspan=1>32,959</td><td rowspan=1 colspan=1>disgusting</td><td rowspan=1 colspan=1>2,388</td><td rowspan=1 colspan=1>average</td><td rowspan=1 colspan=1>3,609</td></tr><tr><td rowspan=1 colspan=1>worth</td><td rowspan=1 colspan=1>22,831</td><td rowspan=1 colspan=1>nothing like</td><td rowspan=1 colspan=1>2,193</td><td rowspan=1 colspan=1>expiration date</td><td rowspan=1 colspan=1>3,263</td></tr><tr><td rowspan=1 colspan=1>great product</td><td rowspan=1 colspan=1>9,269</td><td rowspan=1 colspan=1>never buy</td><td rowspan=1 colspan=1>1,626</td><td rowspan=1 colspan=1>not bad</td><td rowspan=1 colspan=1>3,256</td></tr><tr><td rowspan=1 colspan=1>highly recommended</td><td rowspan=1 colspan=1>5,027</td><td rowspan=1 colspan=1>bad taste</td><td rowspan=1 colspan=1>465</td><td rowspan=1 colspan=1>ive tried</td><td rowspan=1 colspan=1>199</td></tr></table>

Table 1: Sentiment Word Counts for Each Class

## 3.3 Top Used Words For Each Sentiment Class

Defining top-used words, including bigrams (pairs of consecutive words), for each sentiment in this project is essential for understanding sentiment-specific patterns and linguistic trends. Bigrams capture contextual nuances, such as “highly recommended” for positive reviews or “poor quality” for negative ones, ofering deeper insights into customer sentiment. This helps identify positive themes like satisfaction (great, love) and negative pain points (bad, disappointed), providing actionable insights into preferences and complaints. While “good” is the most used word on positive reviews, it was “never” for negative reviews and “fine” for neutral reviews as showed in Table 1.

![](images/5bb60dca2b804876c2f38e0ff678cb6ea4332631590f735e39442f04381fcfa2.jpg)  
Figure 4: Word Weights Coding

This step improves data preprocessing, enabling noise reduction and better feature engineering, while enhancing model performance by focusing on meaningful sentiment indicators. Moreover, it supports clear communication through visualizations like word clouds and reinforces findings with transparency, aiding reproducibility, and cross-dataset comparisons. By uncovering key patterns, it bridges raw data and actionable insights, enriching both academic rigor and business applicability.

## 3.3.1 Weights of Words

Word weighting is important because it enables the quantification of sentiment strength in textual data, providing a more nuanced representation of the emotions expressed in the text. In sentiment analysis, the significance of individual words can vary greatly depending on their context, polarity, and intensity. By assigning appropriate weights to words based on their sentiment value, we can better capture the underlying emotional tone of the content. In this study, we utilized VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based sentiment analysis tool, to assign sentiment weights to words (Figure 4). VADER evaluates the sentiment of words by considering their inherent valence (positive or negative orientation) as well as contextual factors such as punctuation and capitalization, which contribute to the intensity of sentiment. By applying VADER’s word weights, we ensure a more accurate representation of sentiment, where positive words receive higher positive weights, negative words are given negative weights, and neutral words are assigned a value of zero. This method enhances the precision of sentiment classification models, ensuring that the analysis captures both the strength and direction of sentiment, which is particularly valuable in applications such as customer review analysis and social media monitoring.

<table><tr><td rowspan=1 colspan=2>Positive</td><td rowspan=1 colspan=2>Negative</td><td rowspan=1 colspan=2>Neutral</td></tr><tr><td rowspan=1 colspan=1>Word</td><td rowspan=1 colspan=1>Weight</td><td rowspan=1 colspan=1>Word</td><td rowspan=1 colspan=1>Weight</td><td rowspan=1 colspan=1>Word</td><td rowspan=1 colspan=1>Weight</td></tr><tr><td rowspan=1 colspan=1>highly recommended</td><td rowspan=1 colspan=1>0.8534</td><td rowspan=1 colspan=1>never buy</td><td rowspan=1 colspan=1>-0.93</td><td rowspan=1 colspan=1>not bad</td><td rowspan=1 colspan=1>0.431</td></tr><tr><td rowspan=1 colspan=1>love</td><td rowspan=1 colspan=1>0.6369</td><td rowspan=1 colspan=1>bad taste</td><td rowspan=1 colspan=1>-0.5423</td><td rowspan=1 colspan=1>okay</td><td rowspan=1 colspan=1>0.2263</td></tr><tr><td rowspan=1 colspan=1>great</td><td rowspan=1 colspan=1>0.6249</td><td rowspan=1 colspan=1>bad</td><td rowspan=1 colspan=1>-0.5423</td><td rowspan=1 colspan=1>fine</td><td rowspan=1 colspan=1>0.2023</td></tr><tr><td rowspan=1 colspan=1>great product</td><td rowspan=1 colspan=1>0.6249</td><td rowspan=1 colspan=1>disgusting</td><td rowspan=1 colspan=1>-0.5267</td><td rowspan=1 colspan=1>average</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>delicious</td><td rowspan=1 colspan=1>0.5719</td><td rowspan=1 colspan=1>never</td><td rowspan=1 colspan=1>-0.5234</td><td rowspan=1 colspan=1>normal</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>perfect</td><td rowspan=1 colspan=1>0.5719</td><td rowspan=1 colspan=1>worse</td><td rowspan=1 colspan=1>-0.4767</td><td rowspan=1 colspan=1>grocery store</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>good</td><td rowspan=1 colspan=1>0.4404</td><td rowspan=1 colspan=1>disappointed</td><td rowspan=1 colspan=1>-0.47</td><td rowspan=1 colspan=1>expiration date</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>not bad</td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1>awful</td><td rowspan=1 colspan=1>-0.4588</td><td rowspan=1 colspan=1>first time</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>well</td><td rowspan=1 colspan=1>0.2732</td><td rowspan=1 colspan=1>taste like</td><td rowspan=1 colspan=1>-0.2852</td><td rowspan=1 colspan=1>ive tried</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>worth</td><td rowspan=1 colspan=1>0.2263</td><td rowspan=1 colspan=1>nothing like</td><td rowspan=1 colspan=1>-0.2755</td><td rowspan=1 colspan=1>even though</td><td rowspan=1 colspan=1>0.0</td></tr></table>

Table 2: Sentiment Word Weights for Each Class

Neutral sentiment words like “not bad” (0.431), “okay” (0.2263), and “fine” (0.2023) are assigned non-zero weights because they carry mild positive connotations despite being categorized as neutral, as can be seen in Table 2. These words suggest a slightly favorable or acceptable tone, with “not bad” implying mild positivity, “okay” indicating satisfactory acceptance, and “fine” reflecting adequacy. The assigned weights capture these subtle nuances, helping the model recognize the slight sentiment bias they contribute.

## 3.3.2 Preprocessing

The preprocessing stage represents a fundamental step in NLP workflows, serving as the foundation for accurate and reliable sentiment analysis. Its primary objective is to transform raw and unstructured textual data into a clean, standardized, and machine-readable format suitable for model training and evaluation. Since raw customer reviews often contain noise such as spelling variations, redundant characters, and non-textual artifacts, preprocessing is essential to enhance data quality and analytical consistency.

In this study, a comprehensive preprocessing pipeline was implemented (Figure 5) to ensure the dataset’s uniformity and analytical readiness. The process began with tokenization, which segmented text into individual words or tokens, forming the basic input units for feature extraction and model training. Subsequently, duplicate and irrelevant entries were removed to eliminate redundancy and enhance dataset integrity. Punctuation marks, URLs, hashtags, emojis, and stop words were systematically removed or replaced using carefully designed cleaning functions. All text data were then converted to lowercase to ensure case-insensitive consistency across the corpus.

Further linguistic normalization was achieved through the application of stemming and lemmatization, both of which reduced words to their base or root forms, minimizing vocabulary size and improving generalization across linguistic variants. Regular expressions (RegEx) were extensively employed throughout the process to identify and filter specific text patterns, such as special characters and formatting inconsistencies.

As a result of these preprocessing operations-including noise removal, text normalization, and data quality refinement-the dataset was reduced from its original 568,454 reviews to 395,417 cleaned and usable entries, consisting of 356,823 positive, 32,312 negative, and 6,012 neutral reviews and also balancing techniques applied. (Figure 6 to 9) This substantial reduction underscores the importance of rigorous text cleaning in eliminating noisy and unreadable content. The final preprocessed dataset achieved a higher level of clarity, relevance, and interpretability, thus providing a solid foundation for the subsequent feature extraction, model training, and evaluation stages.

![](images/8b1a93cafd219ab5aed3109a66d461350cba0e794701bc7e63e78c6af3a020eb.jpg)  
Figure 5: Preprocessing Coding

## 3.3.3 Data Balancing Methods

The dataset exhibited class imbalance, with the majority class significantly outnumbering the minority classes. To address this, random oversampling was performed, increasing the minority classes (neu. and neg.) to create a balanced dataset of 1,070,469 reviews, as illustrated in Figure 2. To validate the efectiveness of this technique and mitigate bias, models were tested on undersampled subsets (25%, 50%, and 75%) of the dataset, ensuring that oversampling did not artificially inflate performance.

![](images/93bc812dc126aef20c37278502c8424c569b75a4aa85a6ca977fae86c34bff7a.jpg)  
Figure 6: Tokenization Coding

Table 3: Text preprocessing in the machine learning and deep learning pipelines.
<table><tr><td>Step</td><td>ML pipeline</td><td>DL pipeline</td></tr><tr><td>Lowercasing</td><td>Yes</td><td>Yes</td></tr><tr><td>Removal of URLs, men- tions, hashtags, digits,</td><td>Yes</td><td>HTML tags and non-alphabetic characters removed</td></tr><tr><td>punctuation Stopword removal</td><td>Yes (NLTK English list)</td><td>No</td></tr><tr><td>Stemming / lemmatiza- No tion</td><td></td><td>No</td></tr><tr><td>Vectorization</td><td>BoW or TF-IDF, 5000 features</td><td>Integer sequences, padded to 100</td></tr></table>

## 3.3.4 Comparing Various Amount of the Dataset for Accuracy

Oversampling was applied to balance the unbalanced dataset. In case the oversampling is not done correctly, various problems may arise, such as overfitting, noise amplification, biased and skewed models, unrealistic data, computational problems, or misleading model evaluation, which negatively afect the performance of the model and the reliability of the results. Therefore, testing the oversampled data is necessary to make sure the data has been trained correctly. To achieve this, the model was tested on undersampled subsets of the dataset, including 25%, 50%, and 75% of the data. This approach allows for a more thorough evaluation of whether the oversampling process preserved the integrity of the data and did not artificially inflate performance.

![](images/13d2a57a67c62ead757c9109775d16bad781967f9b329ba7e2e38545241999cc.jpg)  
Figure 7: Over Sampling

![](images/26f39a411682e3c914c5a860fc9f093f83b73dbc08592dc4a7928661825a028e.jpg)

Figure 8: Under Sampling  
![](images/365ba2650c80f291b3689124000f8556db6add5a1e7db8b4074b341c4474bd57.jpg)  
Figure 9: Sampling to %75 of the Data

In summary, consistent model performance across undersampled datasets validates that the oversampling process was done correctly and did not distort the dataset’s integrity or artificially enhance accuracy. These tests suggest the oversampling process preserved the integrity of the data without introducing the possible issues that might have occurred in the oversampled dataset.

<table><tr><td rowspan=1 colspan=1>SentimentsDataset Sizes</td><td rowspan=1 colspan=1>Positive</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>Negative</td><td rowspan=1 colspan=1>Total</td></tr><tr><td rowspan=1 colspan=1>Imbalanced Dataset</td><td rowspan=1 colspan=1>356,823</td><td rowspan=1 colspan=1>32,312</td><td rowspan=1 colspan=1>6,012</td><td rowspan=1 colspan=1>395,147</td></tr><tr><td rowspan=1 colspan=1>%25 of the Dataset</td><td rowspan=1 colspan=1>32,928</td><td rowspan=1 colspan=1>32,928</td><td rowspan=1 colspan=1>32,928</td><td rowspan=1 colspan=1>98,784</td></tr><tr><td rowspan=1 colspan=1>%50 of the Dataset</td><td rowspan=1 colspan=1>65,857</td><td rowspan=1 colspan=1>65,857</td><td rowspan=1 colspan=1>65,857</td><td rowspan=1 colspan=1>197,571</td></tr><tr><td rowspan=1 colspan=1>%75 of the Dataset</td><td rowspan=1 colspan=1>98,786</td><td rowspan=1 colspan=1>98,786</td><td rowspan=1 colspan=1>98,786</td><td rowspan=1 colspan=1>296,358</td></tr><tr><td rowspan=1 colspan=1>%100 of the Dataset</td><td rowspan=1 colspan=1>131,715</td><td rowspan=1 colspan=1>131,715</td><td rowspan=1 colspan=1>131,715</td><td rowspan=1 colspan=1>395,145</td></tr><tr><td rowspan=1 colspan=1>Oversampled Dataset</td><td rowspan=1 colspan=1>356,823</td><td rowspan=1 colspan=1>356,823</td><td rowspan=1 colspan=1>356,823</td><td rowspan=1 colspan=1>1,070,469</td></tr></table>

Table 4: Various Percentages of the Dataset

## 4 Machine Learning Models

In this study, five machine learning models -RFC, LRC, BNB, L-SVC, MNB- were applied to evaluate the performance of sentiment analysis on the Amazon customer review dataset. Two feature extraction techniques, BoW and TF-IDF, were utilized to transform the textual data into numerical representations suitable for model training and evaluation. The analysis focuses on a comparative assessment of the top three models for each feature extraction method, as well as a comprehensive comparison of the five models and the two feature extraction approaches. The evaluation metrics - accuracy, precision, recall and F1 score - were calculated to measure the overall performance of each model. Furthermore, the metrics were further analyzed at a class level (positive, negative, and neutral) for every model and feature extraction method, providing a detailed insight into their efectiveness across diferent sentiment categories. This multidimensional evaluation facilitates a strong understanding of the interaction between model architectures and feature extraction techniques in sentiment classification tasks. The overall operational workflow is illustrated in Figure 10,

![](images/10e360ab5f679e0ca56a075d646209770d6576a0f47b6a3fe66a080e3662854c.jpg)  
Figure 10: Operational Flow Overview for Machine Learning

## 4.1 General Aspects of Machine Learning

## 4.1.1 Feature Extraction

The evaluation considered both overall performance and class-level metrics, allowing for a comprehensive comparison of the top three performing models for each feature extraction method, as well as a broader assessment of all five models across both techniques. Metrics including accuracy, precision, recall, and F1 score were calculated to quantify model efectiveness, providing a multidimensional understanding of how model architectures interact with feature representations in sentiment analysis tasks.

For BoW, scikit-learn’s CountVectorizer was employed to convert textual reviews into numerical vectors. The vectorizer preprocesses the text by tokenizing into words, removing single-character tokens, normalizing accented characters, and extracting both unigrams and bigrams (n-gram range = 1–2). To maintain computational eficiency and reduce sparsity, the feature set was limited to the 5000 most frequent tokens. The selection of 5000 features strikes a balance between capturing suficient vocabulary for meaningful sentiment analysis and ensuring manageable computational complexity. The use of unigrams and bigrams allows the model to capture both individual word contributions and common two-word phrases, which improves the representation of context in reviews. The resulting BoW representations

# BOW   
from sklearn.feature extraction.text import CountVectorizer   
import pickle   
count vectorizer = CountVectorizer(   
analyzer = 'word',   
strip\_accents='unicode',   
token\_pattern = r'\w{2,}',   
ngram\_range = (1,2),   
max\_features = 5000) # Extract 5000 Features   
train\_text = preprocessed\_dataset['Text']   
count\_vectorizer = count\_vectorizer.fit(train\_text)   
vectorizer\_filename = 'trained\_count\_vectorizer.pkl   
with open(vectorizer\_filename, 'wb') as f:   
pickle.dump(count\_vectorizer, f)

## Figure 11: BoW Feature Extraction Coding

![](images/a9c7bb360976bc85850d364510da83c43b1fecd2e48fc8fbece2ead4a0c64a1a.jpg)  
Figure 12: TF-IDF Feature Extraction Coding

served as input to the machine learning models. To ensure reproducibility, the trained BoW vectorizer was saved using Python’s pickle module, enabling consistent feature extraction for future experiments and applications.(Figure 11)

For TF-IDF, scikit-learn’s TfidfVectorizer was applied to account for the relative importance of terms across the corpus. Similar preprocessing steps were applied, including tokenization, removal of single-character tokens, accent normalization, and extraction of unigrams and bigrams. The 5000 most frequent terms were selected to form the feature set, and term weights were computed based on their TF-IDF values, capturing both term frequency and inverse document frequency information. This approach ensures that common but less informative words are weighted lower, while rare but sentiment-relevant words receive higher importance. The resulting TF-IDF weighted vectors were then used as input for the same set of machine learning models. The trained TF-IDF vectorizer was also saved for reproducibility, ensuring that future studies or real-time applications can employ the exact same

![](images/c09ef984a02bd40c04bf803ba3681b2796240e7b6722d20fd069903f5e74b119.jpg)  
Figure 13: Word Cloud Coding

feature mapping.(Figure 12)

This dual-feature extraction strategy enabled a thorough investigation of the comparative strengths of BoW and TF-IDF representations, providing insights into their impact on model performance across diferent sentiment categories and guiding the selection of the most efective combination of model and feature representation for large-scale sentiment analysis.

## 4.1.2 Word Cloud

The Word Cloud method is a visual representation of textual data where the frequency or importance of words within a dataset is depicted using varying font sizes. Larger words in the word cloud indicate higher frequency or relevance, while smaller words represent less frequent terms. The coding structure is illustrated in Figure 13, and the resulting word clouds are shown in Figures 14–15-16. In the context of a large-scale sentiment analysis project on the Amazon customer review dataset, this method plays a crucial role in providing an intuitive and immediate understanding of the most frequently used terms across millions of reviews. It helps identify prominent themes, recurring terms, and potential sentimentindicative keywords, such as “excellent,” “disappointed,” “fast,” or “poor,” which may otherwise be overlooked in raw data. This is particularly useful in studies dealing with massive datasets, as word clouds ofer a quick way to visualize the general sentiment and focus of the reviews. Moreover, they assist in uncovering key patterns and variations in customer sentiment, providing insights that can guide further preprocessing steps, feature engineering, and model development. By revealing the linguistic trends and customer priorities embedded in the data, word clouds serve as a valuable exploratory tool to complement deeper analytical methodologies.

In this study, we used the Word Cloud method for every class separately as positive, negative, and neutral, as shown in Figures 14, 15, and 16. Furthermore, we have used such methods as top-used words and weights for each word to describe the aspects of the project, enhance the analysis for the dataset, and explore new features to serve the sentiment decisions. Methods that are mentioned will be shown in the next sections.

## 4.2 Confusion Matrix

The Confusion Matrix, is a crucial evaluation tool for classification tasks, providing a detailed breakdown of correct and incorrect predictions across all classes. For a 3-class sentiment analysis (positive, neutral, negative), it organizes true labels as rows and predicted labels as columns in a 3x3 grid. Diagonal cells represent correct classifications, while ofdiagonal cells indicate misclassifications, such as labeling a positive review as neutral.

![](images/68eb384a7a63ba9de4a7164ad21508705af188b7e98463c2a6aa805b32775e9a.jpg)  
Figure 14: Positive Sentiment Word Cloud

![](images/c1a4d520249ce7aef0cba39a5682b12cc845b57bbc553201099a464e58568f66.jpg)

Figure 15: Negative Sentiment Word Cloud  
![](images/93a13a1f615f95cee2b77dfc89262f9f7debc8f94eddaf7c2376631b056482f3.jpg)  
Figure 16: Neutral Sentiment Word Cloud

This method is essential for identifying model weaknesses and calculating metrics like precision, recall, and F1-score for each class. It ofers deeper insights into performance, helping refine the model and address challenges like overlapping features or class imbalance. Figure 17 displays a sample confusion matrix and Figure 18 contains the coding, further details will be provided in the subsequent sections.

## 4.3 Presentation of Machine Learning Models

Five machine learning models-RFC, LRC, BNB, L-SVC, and MNB-were implemented to classify sentiment within the dataset. To prepare the textual data for these models, two feature extraction methods, TF-IDF and BoW, were employed. Analysis of the results showed that the BoW method provided superior performance compared to TF-IDF. As a result, the primary focus is on comparing the first three models trained using the BoW method. For improved visualization, the model metrics are presented in Figure 19.

Performance evaluation is conducted using confusion matrices and detailed metrics, including precision, recall, F1 score, and accuracy, for each sentiment class. This approach ensures a comprehensive comparison of the models and their ability to classify sentiments efectively. The best-performing model is then highlighted, and its results are analyzed to provide insights into its strengths. Additionally, the comparison between the feature extraction methods will be elaborated in the next section to ofer a broader perspective on their influence on model performance.

![](images/3d943352132906ccbbe52b64f1c5dcf03bc3c34f9b6cd5913b9d420ea5914559.jpg)

Figure 17: Sample Confusion Matrix  
![](images/f3309df2ac02760c28d16bf1cbf86b7c32223c2b00aa36dfae7155bcf714d00b.jpg)  
Figure 18: Confusion Matrix Coding

## 4.3.1 Random Forest Classifier (RFC)

The RFC is an ensemble machine learning algorithm designed to enhance classification performance by constructing multiple decision trees during training and aggregating their predictions. This approach reduces overfitting and improves model robustness by combining predictions from various trees, each trained on random subsets of features and data. Its ability to handle high-dimensional data, manage noisy features, and accommodate missing values makes it a widely adopted method for complex classification tasks, such as sentiment analysis.

Bootstrap Sampling (Random Data Selection): Each of the 5 trees is trained on a random sample of the training data with replacement. This introduces variability and reduces overfitting.

Random Feature Selection at Each Split:

$$
m = \sqrt { M }\tag{1}
$$

where m is the number of randomly selected features out of M total features.

Table 5: Parameters of the machine learning models. Unless stated, values are scikit-learn defaults.
<table><tr><td>Parameter</td><td>RFC</td><td>LRC</td><td>L-SVC</td><td>BNB</td></tr><tr><td>Implementation Number of trees</td><td>RandomForestClassifier n_estimators=5</td><td>LogisticRegression</td><td>LinearSVC</td><td>BernoulliNB</td></tr><tr><td>Max iterations Regularization</td><td></td><td>max_iter=100</td><td>max_iter=1000</td><td>α=1.0</td></tr><tr><td>Loss / solver</td><td>Gini impurity</td><td>L2, C=1.0 lbfgs</td><td>L2, C=1.0 squared hinge</td><td></td></tr><tr><td>Class prior</td><td></td><td></td><td></td><td>training-split proporti</td></tr><tr><td>Binarization Fit prior</td><td></td><td></td><td></td><td>binarize=0.0</td></tr><tr><td>Class weights (oversampled set)</td><td></td><td></td><td></td><td>True</td></tr><tr><td></td><td>none</td><td>none</td><td>none</td><td>none</td></tr><tr><td>Class weights (imbalanced set)</td><td>balanced</td><td>balanced</td><td>balanced</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>none</td></tr><tr><td>Random seed</td><td>random_state=0</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Feature selection per split</td><td>√M</td><td></td><td></td><td></td></tr></table>

Feature extraction. BoW: CountVectorizer, unigrams and bigrams (ngram range=(1,2)), token pattern=\w{2,}, strip accents=unicode, max features=5000. TF-IDF: TfidfVectorizer with the same settings; max features=5000 in the oversampled run and 1000 in the imbalanced run. Data split. 80/20 train/test split, random state=2020, not stratified.

Gini Impurity (Node Splitting):

$$
G ( t ) = 1 - \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 }\tag{2}
$$

where $p _ { k }$ is the proportion of samples belonging to class k at node t.

Majority Voting (Final Prediction):

$$
\hat { y } = \mathrm { m o d e } \left( \sum _ { i = 1 } ^ { N } h _ { i } ( x ) \right)\tag{3}
$$

where $h _ { i } ( x )$ is the prediction from the i-th decision tree and N is the total number of trees.   
$\hat { y }$ is the predicted class label.

In this study, the RFC emerged as the most successful and accurate model, utilizing the BoW feature extraction method. The results from applying this model to the Amazon review dataset demonstrate its efectiveness, achieving an overall accuracy of 99%. The model performed exceptionally across all sentiment classes, with precision, recall, and F1-scores exceeding 0.96 for each class. Analysis of the confusion matrix reveals perfect classification of neutral sentiments (71,300 instances) and minimal misclassifications for negative and positive sentiments. The parameter “support” in parameter tables, as Figure 20, represents the total number of instances in the dataset that belong to each sentiment class and all dataset. Out of 71,628 negative samples, only 22 were misclassified, while 2,718 positive samples were incorrectly identified as negative, and 206 as neutral (Figure 21). These results highlight the model’s strength in accurately classifying neutral and negative sentiments, while minor misclassifications in positive sentiments suggest slight feature overlaps. Overall, the RFC proved highly reliable, demonstrating its capacity for large-scale sentiment classification with complex datasets.

![](images/c1ad132c76b3b6d58eb176df0748c1af2cca1ac3786edffb50d745862665766a.jpg)  
Figure 19: Model Metrics with BoW

<table><tr><td colspan="2">precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>Negative</td><td>0.96</td><td>1.00</td><td>0.98</td><td>71628</td></tr><tr><td>Neutral</td><td>1.00</td><td>1.00</td><td>1.00</td><td>71300</td></tr><tr><td>Positive</td><td>1.00</td><td>0.96</td><td>0.98</td><td>71166</td></tr><tr><td>accuracy</td><td></td><td></td><td>0.99</td><td>214094</td></tr><tr><td>macro avg</td><td>0.99</td><td>0.99</td><td>0.99</td><td>214094</td></tr><tr><td>weighted avg</td><td>0.99</td><td>0.99</td><td>0.99</td><td>214094</td></tr></table>

Figure 20: Parameters for Each Class in RFC

![](images/1479af974cab6e6e8d260238a303f5566b7476349b8547383907a5a6af894c79.jpg)  
Figure 21: RFC - Confusion Matrix

## 4.3.2 Logistic Regression Classifier (LRC)

The LRC is a widely used algorithm in machine learning that is particularly efective for binary and multiclass classification tasks. It models the probability of a class by applying the logistic function to a linear combination of input features, ensuring the output probabilities remain between 0 and 1. This method is computationally eficient and interpretable, making it well-suited for large-scale datasets and sentiment analysis applications. Furthermore, LRC can handle high-dimensional data and produce probabilistic outputs, which are advantageous for understanding the confidence levels of predictions.

Model Initialization:

The Logistic Regression model starts by initializing the weight vector w and bias b.

Model Training:

During training, the model learns the best weights w and bias $b$ that minimize the Cross-Entropy Loss function:

$$
\mathbf { w }  \mathbf { w } - \eta \nabla L ( \mathbf { w } )\tag{4}
$$

where $\eta$ is the learning rate (step size), and $\nabla L ( \mathbf { w } )$ is the gradient of the loss function, which indicates the direction to adjust the weights to reduce the loss.

Prediction Using Sigmoid Function:

After training, the model uses the Sigmoid Function to calculate the probability of a class for a given input x:

$$
P ( y = 1 \mid \mathbf { x } ) = \sigma ( \mathbf { w } ^ { \top } \mathbf { x } + b ) = \frac { 1 } { 1 + e ^ { - ( \mathbf { w } ^ { \top } \mathbf { x } + b ) } }\tag{5}
$$

where $\mathbf { w } ^ { \top } \mathbf { x } + b$ is the linear combination of input features and their corresponding weights.   
The output is a probability between 0 and 1.

Decision Rule:

The class is predicted based on the probability:

$$
{ \hat { y } } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } P ( y = 1 \mid \mathbf { x } ) \geq 0 . 5 } \\ { 0 , } & { { \mathrm { i f ~ } } P ( y = 1 \mid \mathbf { x } ) < 0 . 5 } \end{array} \right. }\tag{6}
$$

where $\hat { y }$ is the predicted class label.

When applied to the Amazon review dataset, the LRC model achieved an overall accuracy of 90%. The precision, recall, and F1-scores for negative, neutral, and positive sentiments were 0.88, 0.91, and 0.91, respectively, with the highest performance observed in the neutral class (F1-score of 0.94). The confusion matrix indicates that out of 71,628 negative reviews, 5,721 were misclassified as positive, and 4,577 as neutral. Similarly, among 71,166 positive reviews, 6,487 were misclassified as negative, and 2,052 as neutral, as Figure 22-23. Neutral reviews displayed minimal misclassification, with 2,092 and 145 instances misclassified as negative and positive, respectively. While the model demonstrates robust performance, the misclassifications highlight areas for potential improvement, such as addressing overlapping features or imbalanced class distributions.

## 4.3.3 Linear Support Vector Classification (L-SVC)

L-SVC is a machine learning algorithm well-suited for text classification tasks due to its eficiency with high-dimensional datasets. It uses a linear decision boundary to separate classes and maximizes the margin between data points, making it robust and scalable for large datasets. L-SVC is particularly advantageous for sentiment analysis, as it eficiently handles sparse data representations such as BoW and TF-IDF feature extraction methods.

Decision Boundary: The decision boundary is the line (or hyperplane) that separates the data into two classes. It is determined by the equation:

$$
f ( x ) = w ^ { \top } x + b\tag{7}
$$

<table><tr><td></td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>Negative</td><td>0.88</td><td>0.86</td><td>0.87</td><td>71628</td></tr><tr><td>Neutral</td><td>0.91</td><td>0.97</td><td>0.94</td><td>71300</td></tr><tr><td>Positive</td><td>0.91</td><td>0.88</td><td>0.90</td><td>71166</td></tr><tr><td>accuracy</td><td></td><td></td><td>0.90</td><td>214094</td></tr><tr><td>macro avg</td><td>0.90</td><td>0.90</td><td>0.90</td><td>214094</td></tr><tr><td>weighted avg</td><td>0.90</td><td>0.90</td><td>0.90</td><td>214094</td></tr></table>

Figure 22: Parameters for Each Class in LRC

![](images/7a8116c115e7339358ebe7267fcf5c01cac80c410ff946d89a4d53b8d5cbdf5b.jpg)  
Figure 23: LRC - Confusion Matrix

where w is the weight vector, x is the feature vector, and b is the bias term.

The class prediction is:

$$
{ \hat { y } } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } f ( x ) > 0 , } \\ { - 1 } & { { \mathrm { i f ~ } } f ( x ) < 0 } \end{array} \right. }\tag{8}
$$

Margin Maximization: To improve the classification performance, we maximize the margin between the two classes:

$$
y _ { i } ( w ^ { \top } x _ { i } + b ) \geq 1 , \quad \forall i\tag{9}
$$

The margin is given by:

$$
\mathrm { M a r g i n } = { \frac { 2 } { \| w \| } }\tag{10}
$$

Hinge Loss Function: The hinge loss function penalizes misclassified points or points close to the margin. It is defined as:

$$
L ( w ) = \sum _ { i = 1 } ^ { N } \operatorname* { m a x } ( 0 , 1 - y _ { i } ( w ^ { \top } x _ { i } + b ) ) + \frac { \lambda } { 2 } \| w \| ^ { 2 }\tag{11}
$$

where λ is the regularization parameter that prevents overfitting.

L-SVC was applied to the Amazon review dataset with BoW as the feature extraction method. The model demonstrated strong classification performance, achieving an overall accuracy of 90%. The evaluation metrics highlight the model’s balanced efectiveness across all sentiment classes. The precision, recall, and F1-scores for negative, neutral, and positive sentiments were consistently high, exceeding 0.85 in all cases. Specifically, neutral sentiments showed the highest precision (0.92) and recall (0.97), leading to an F1-score of 0.94. Negative and positive sentiments achieved F1-scores of 0.87 and 0.90, respectively.

The confusion matrix further illustrates the model’s reliability. Neutral sentiments were classified with the highest accuracy, with only 1,971 misclassifications out of 71,300 instances.

Negative sentiments saw 6,491 misclassified as positive and 4,277 as neutral. For positive sentiments, 6,123 samples were misclassified as negative, while 1,969 were labeled neutral, as Figure 24-25. These results emphasize L-SVC’s robustness in sentiment classification, particularly for neutral and positive classes, while minor overlaps in feature representation influenced the misclassification of negative sentiments.

<table><tr><td></td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>Negative</td><td>0.88</td><td>0.85</td><td>0.87</td><td>71628</td></tr><tr><td>Neutral</td><td>0.92</td><td>0.97</td><td>0.94</td><td>71300</td></tr><tr><td>Positive</td><td>0.90</td><td>0.89</td><td>0.90</td><td>71166</td></tr><tr><td>accuracy</td><td></td><td></td><td>0.90</td><td>214094</td></tr><tr><td>macro avg</td><td>0.90</td><td>0.90</td><td>0.90</td><td>214094</td></tr><tr><td>weighted avg</td><td>0.90</td><td>0.90</td><td>0.90</td><td>214094</td></tr></table>

Figure 24: Parameters for Each Class in L-SVC

![](images/24444b6dd699abd1fc50704b46e6aa81603df105c14cd2a34e6806f1fdcc82db.jpg)  
Figure 25: L-SVC Confusion Matrix

## 4.3.4 Bernoulli Naive Bayes (BNB)

BNB is a probabilistic classification algorithm commonly used for text classification tasks, especially when the input features are binary. In this model, each feature represents whether a word appears in a document (1) or not (0). This makes BNB particularly well-suited for Bag-of-Words representations where the frequency of a word is less important than its presence. The Bernoulli assumption simplifies computation and results in eficient classification performance, especially for large and sparse datasets such as sentiment analysis on customer reviews. Bernoulli Probability Model: The likelihood of a document belonging to class c is computed as:

$$
P ( x _ { i } = 1 \mid c ) = \theta _ { i c } , \quad P ( x _ { i } = 0 \mid c ) = 1 - \theta _ { i c }\tag{12}
$$

where $\theta _ { i c }$ is the probability that feature $x _ { i }$ occurs in class c.

Posterior Class Prediction:

$$
{ \hat { y } } = \arg \operatorname* { m a x } _ { c \in C } P ( c ) \prod _ { i = 1 } ^ { n } P ( x _ { i } \mid c )\tag{13}
$$

where P(c) is the prior probability of class c.

When applied to the Amazon review dataset, the BNB classifier achieved an overall accuracy of 76%. The model demonstrated reasonable performance across all sentiment categories, with F1-scores of 0.70 (negative), 0.81 (neutral), and 0.76 (positive). Neutral sentiments were identified with the highest recall (0.92), indicating the model’s strong ability to detect neutral reviews, likely because neutral text often contains fewer emotionally charged words and aligns well with the Bernoulli representation.

Negative reviews showed the lowest recall (0.64), revealing that a substantial portion of negative instances were predicted as either neutral or positive due to overlapping vocabulary. Positive sentiment classification was moderately strong with a precision of 0.81 and recall of 0.72. The macro and weighted averages (all 0.76) further indicate that the classifier performed consistently across the three sentiment classes.

Overall, the BNB model provides a fast and interpretable baseline for sentiment analysis. However, its binary feature formulation limits its ability to capture nuanced textual patterns found in longer customer reviews, which explains the moderate overall accuracy of 76%, along with class-level F1-scores of 0.70 (negative), 0.81 (neutral), and 0.76 (positive). Compared to other classical and ensemble models used in this study, BNB underperforms noticeably. The RFC achieved an exceptionally high accuracy of 99% with F1-scores exceeding 0.96 across all sentiment classes, demonstrating far greater robustness. Similarly, both L-SVC and LRC achieved 90% accuracy, with F1-scores of 0.87–0.94 across classes-substantially outperforming the BNB model, as Figure 26-27. These diferences highlight the limitations of binary feature modeling in capturing sentiment-rich language compared to linear and ensemble approaches that more efectively leverage the full representation of textual features.

<table><tr><td colspan="5"></td></tr><tr><td></td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>Negative</td><td>0.77</td><td>0.64</td><td>0.70</td><td>71628</td></tr><tr><td>Neutral</td><td>0.72</td><td>0.92</td><td>0.81</td><td>71300</td></tr><tr><td>Positive</td><td>0.81</td><td>0.72</td><td>0.76</td><td>71166</td></tr><tr><td>accuracy</td><td></td><td></td><td>0.76</td><td>214094</td></tr><tr><td>macro avg</td><td>0.77</td><td>0.76</td><td>0.76</td><td>214094</td></tr><tr><td>weighted avg</td><td>0.77</td><td>0.76</td><td>0.76</td><td>214094</td></tr></table>

Figure 26: Parameters for Each Class in BNB

![](images/957c33cf15aac7b47002774186cd7427c1559f375f1afb736fa8242c7abb052f.jpg)  
Figure 27: BNB Confusion Matrix

## 4.3.5 Multinomial Naive Bayes (MNB)

MNB is a widely used probabilistic classifier for text analysis and is particularly efective when features represent word counts or term frequencies. Unlike BNB, which considers only the presence or absence of words, the MNB model leverages the frequency of each word in a document. This makes it especially powerful for sentiment analysis tasks using BoW or TF-IDF features, where word occurrence rates carry important semantic information.

Multinomial Likelihood Function:

$$
P ( \mathbf { x } \mid c ) = { \frac { ( \sum _ { i } x _ { i } ) ! } { \prod _ { i } x _ { i } ! } } \prod _ { i = 1 } ^ { n } \theta _ { i c } ^ { x _ { i } }\tag{14}
$$

where $x _ { i }$ is the count of word i, and $\theta _ { i c }$ is the probability of word i appearing in class c. Posterior Prediction Rule:

$$
\hat { y } = \arg \operatorname* { m a x } _ { c \in C } ; P ( c ) \prod _ { i = 1 } ^ { n } \theta _ { i c } ^ { x _ { i } }\tag{15}
$$

The MNB classifier outperformed the Bernoulli variant, achieving an overall accuracy of 79% on the Amazon review dataset. The model demonstrated strong performance across all classes, with F1-scores of 0.75 (negative), 0.85 (neutral), and 0.79 (positive). Neutral sentiments achieved the highest precision (0.88) and strong recall (0.82), indicating the model’s ability to capture the typical structure of neutral reviews more efectively.

Positive sentiments were also classified well, with a recall of 0.81 and an F1-score of 0.79, reflecting the model’s capability to recognize emotionally charged vocabulary when evaluated using word frequency information. Negative sentiment classification showed moderate performance, with balanced precision and recall values (0.74 and 0.75), suggesting some overlap between negative and neutral textual patterns but a stable overall predictive ability.

Both macro and weighted averages (precision: 0.80, recall: 0.79, F1-score: 0.80) highlight the model’s balanced performance across the dataset. The improvement of MNB over BNB can be attributed to its ability to leverage term frequencies, which carry richer contextual and semantic meaning.

Overall, the MNB model demonstrates strong suitability for large-scale sentiment analysis tasks, ofering a good balance of accuracy, computational eficiency, and interpretability. Its overall accuracy of 79%, along with F1-scores of 0.75 (negative), 0.85 (neutral), and 0.79 (positive), indicates stable and reliable performance across sentiment classes. Nevertheless, when compared to more advanced models in this study, such as RFC, L-SVC, and LRC, the MNB model is clearly outperformed. The RFC achieved 99% accuracy and near-perfect F1- scores above 0.96, representing a significant performance margin over MNB. Likewise, L-SVC and LRC both reached 90% accuracy, with class-level F1-scores ranging from 0.87 to 0.94, surpassing MNB’s performance in all sentiment categories, as Figure 28-29. These results show that although MNB is efective for text classification, modern linear and ensemble models capture sentiment-related patterns more comprehensively, especially in high-dimensional feature spaces generated by large review datasets.

## 4.4 Comparison of Top 3 Models

RFC uses decision trees with random feature selection and majority voting to reduce variance and model non-linear relationships. LRC applies the logistic or softmax function to estimate class probabilities, optimized by cross-entropy loss. L-SVC maximizes the margin between classes, penalizing misclassifications using the hinge loss. Understanding these mathematical foundations clarifies how each model handles multi-class sentiment analysis.

<table><tr><td colspan="2">precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>Negative</td><td>0.74</td><td>0.75</td><td>0.75</td><td>71628</td></tr><tr><td>Neutral</td><td>0.88</td><td>0.82</td><td>0.85</td><td>71300</td></tr><tr><td>Positive</td><td>0.77</td><td>0.81</td><td>0.79</td><td>71166</td></tr><tr><td>accuracy</td><td></td><td></td><td>0.79</td><td>214094</td></tr><tr><td>macro avg</td><td>0.80</td><td>0.79</td><td>0.80</td><td>214094</td></tr><tr><td>weighted avg</td><td>0.80</td><td>0.79</td><td>0.80</td><td>214094</td></tr></table>

Figure 28: Parameters for Each Class in MNB

![](images/5c53e48240a40904b385e2dfab22810fd0a7d4410eff1c2f2658b1a872c59a92.jpg)  
Figure 29: MNB Confusion Matrix

L-SVC, a margin-maximizing classifier, aims to separate classes by identifying a hyperplane with the largest margin. While this approach works well for binary classification, its adaptation to multi-class problems introduces challenges, particularly in distinguishing between neutral and the more polarized positive and negative sentiments. The model performed admirably for positive and negative classes but showed dificulty in handling the neutral class, likely due to the lack of nuanced feature modeling. Compared, RFC outperformed both LRC and L-SVC in providing a more balanced classification across all three sentiment classes. LRC’s straightforward probabilistic approach and L-SVC’s margin-based separation were efective but exhibited limitations in capturing the subtleties of multi-class sentiment analysis, emphasizing the importance of model selection based on the dataset’s complexity and classification goals.

![](images/3ce7fdf8a5587ff3e31ba6319d4f4a3abbee522cf8600a54164c2a74b595d07a.jpg)  
Figure 30: Metrics of Top 3 Model

## 4.4.1 Classification Reports

Figure 30 visually compares the performance of the RFC, LRC, and L-SVC in a multiclass sentiment analysis task (negative, neutral, positive).

In a confusion matrix, True Positives (TP) are cases where the model correctly predicts the positive class, while True Negatives (TN) are correct predictions of the negative class. False Positives (FP) occur when the model incorrectly labels a negative instance as positive (a false alarm), and False Negatives (FN) occur when the model fails to detect a positive instance and predicts it as negative (a missed detection). These four outcomes form the basis for evaluation metrics like accuracy, precision, recall, and F1-score. Evaluation metrics-Precision, Recall, F1-Score, and Accuracy-provide critical insights into model efectiveness.

Precision is defined as:

$$
P = { \frac { T P } { T P + F P } }\tag{16}
$$

It measures the model’s ability to correctly identify positive instances, reflecting how reliable positive predictions are.

![](images/c7ec6ea33330d49e21064f5de9ff43a0a0ba77ff5ac8cc51c7c1cb9aa7e71805.jpg)  
Figure 31: Classification Report Metrics

Recall is calculated as:

$$
R = { \frac { T P } { T P + F N } }\tag{17}
$$

It evaluates how well the model captures all relevant positive cases, essential for minimizing false negatives.

F1-Score, the harmonic mean of precision and recall, is given by:

$$
F 1 = \frac { 2 \times P \times R } { P + R }\tag{18}
$$

Accuracy measures the overall correctness of the model:

$$
A = { \frac { T P + T N } { T P + T N + F P + F N } }\tag{19}
$$

In Figure 31, RFC demonstrates robust performance across metrics due to its ensemble structure, excelling in capturing complex patterns. LRC shows solid but slightly lower performance, while L-SVC struggles with neutral sentiment classification due to its linear separation approach. This comparison highlights RFC’s superior adaptability for nuanced sentiment classification, making it the most balanced model for this task.[11]

## 4.4.2 Running Time Comparison

L-SVC demonstrated the fastest execution time among the three models, benefiting from its eficient linear computations; however, it achieved the lowest accuracy at 89%. LRC ofered a balance between computational eficiency and predictive performance, with a moderate running time and an accuracy of 90%. In contrast, RFC attained the highest accuracy at 99% but incurred the longest execution time due to its ensemble of decision trees and complex feature evaluations. (Figure 32) Although RFC appears to be the most accurate model, the considerable training time raises concerns about its practicality in time-sensitive applications. Therefore, the trade-of between computational eficiency and predictive performance will be examined in greater detail in the subsequent sections.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Running Time(minutes:seconds)</td></tr><tr><td rowspan=1 colspan=1>Random Forest (RFC)</td><td rowspan=1 colspan=1>16m 55s (1015s)</td></tr><tr><td rowspan=1 colspan=1>Logistic Regression (LRC)</td><td rowspan=1 colspan=1>3m 37s (217s)</td></tr><tr><td rowspan=1 colspan=1>Linear SVC</td><td rowspan=1 colspan=1>1m 15s (75s)</td></tr></table>

Figure 32: Running Times of Top 3 Model

## 4.5 Comparison of Feature Extraction Methods

## 4.5.1 Bag of Words (BoW)

BoW model converts text data into a matrix of word counts, ignoring grammar and word order while retaining word frequency. Each document is transformed into a vector where each element reflects the occurrence of a specific word. This method is straightforward and efective for recognizing general word usage patterns but lacks the ability to capture contextual nuances.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>BoW</td><td rowspan=1 colspan=1>TF-IDF</td></tr><tr><td rowspan=1 colspan=1>Random Forest Classifier (RFC)</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.98</td></tr><tr><td rowspan=1 colspan=1>Logistic Regression Classifier (LRC)</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>Linear Support Vector Classifier (L-SVC)</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=1 colspan=1>Bernoulli Naive Bayes (BNB)</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.74</td></tr><tr><td rowspan=1 colspan=1>Multinomial Naive Bayes (MNB)</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.76</td></tr></table>

Table 6: Feature Extraction Method Accuracy Comparison
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>BoW (Time)</td><td rowspan=1 colspan=1>TF-IDF (Time)</td></tr><tr><td rowspan=1 colspan=1>Random Forest Classifier (RFC)</td><td rowspan=1 colspan=1>16m 55s</td><td rowspan=1 colspan=1>17m 16s</td></tr><tr><td rowspan=1 colspan=1>Logistic Regression Classifier (LRC)</td><td rowspan=1 colspan=1>3m 37s</td><td rowspan=1 colspan=1>3m 36s</td></tr><tr><td rowspan=1 colspan=1>Linear Support Vector Classifier (L-SVC)</td><td rowspan=1 colspan=1>1m 15s</td><td rowspan=1 colspan=1>1m 14s</td></tr><tr><td rowspan=1 colspan=1>Bernoulli Naive Bayes (BNB)</td><td rowspan=1 colspan=1>0m 29s</td><td rowspan=1 colspan=1>0m 29s</td></tr><tr><td rowspan=1 colspan=1>Multinomial Naive Bayes (MNB)</td><td rowspan=1 colspan=1>0m 6s</td><td rowspan=1 colspan=1>0m 6s</td></tr></table>

Table 7: Feature Extraction Method Time Comparison

## 4.5.2 Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF enhances the BoW approach by weighing word frequency against the uniqueness of words across documents. This model down-weights frequently occurring words and highlights rarer, more informative terms. Consequently, TF-IDF ofers a more balanced and contextually rich feature representation, which can be more efective for sentiment classification. Model accuracies and training times can be found in Table 4-5 for better comparison across the feature extraction techniques.

## Comparative Analysis of Feature Extraction Impact

• Bag of Words efectively captures frequent sentiment words, benefiting models like RFC that handle high-dimensional data.

• TF-IDF highlights contextually important words but introduces sparsity, which may hinder simpler models.

• RFC consistently outperformed other models with both methods, showing a slight preference for BoW.

• LRC and L-SVC showed slight declines with TF-IDF due to challenges in handling weighted features.

• Naive Bayes models underperformed with both methods, highlighting their sensitivity to feature representation.

The BoW method produced slightly higher accuracy across most models, suggesting that frequent word occurrence was suficient for diferentiating between positive, negative, and neutral sentiments. Although TF-IDF theoretically provides richer feature representations, its sparsity and weighting complexities slightly hindered model performance, especially for simpler classifiers. This comparison underscores the significance of selecting appropriate feature extraction techniques for optimal sentiment analysis performance.

## 4.6 Best Machine Learning Model Discussion and Summary

<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Feature Extraction</td><td rowspan=1 colspan=1>Performance Metric</td></tr><tr><td rowspan=1 colspan=1>Fastest Model</td><td rowspan=1 colspan=1>Multinomial Naive Bayes</td><td rowspan=1 colspan=1>BoW</td><td rowspan=1 colspan=1>0m 6s</td></tr><tr><td rowspan=1 colspan=1>Most Accurate Model</td><td rowspan=1 colspan=1>Random Forest Classifier</td><td rowspan=1 colspan=1>BoW</td><td rowspan=1 colspan=1>0.99 Accuracy</td></tr><tr><td rowspan=1 colspan=1>Most Efficient Model</td><td rowspan=1 colspan=1>Linear Support Vector Classifier</td><td rowspan=1 colspan=1>BoW</td><td rowspan=1 colspan=1>E = T × A</td></tr></table>

## Table 8: Summary of Best Performing Models by Category

A comprehensive evaluation of the machine learning models was conducted using BoW and TF-IDF feature extraction methods. The models are assessed based on their accuracy, training time, and overall eficiency.

The eficiency parameter, defined as

$$
{ \mathrm { E f f i c i e n c y } } = { \mathrm { T i m e } } \times { \mathrm { A c c u r a c y } } ,
$$

evaluates machine learning models by balancing their predictive performance and computational cost. Accuracy reflects how well the model performs in predictions, while training time measures the resources required for model training. Combining these factors, the efficiency metric provides a practical and holistic measure, highlighting models that achieve high accuracy with minimal computational expense. This balance ensures the selection of models that are not only efective but also resource-eficient, making the metric especially valuable for real-world applications where both performance and speed are critical.

• RFC achieved the highest accuracy for both feature extraction methods. However, it is also the most time-consuming, requiring 16m 55s for BoW and 17m 16s for TF-IDF. This is nearly five times longer than the LRC.

• LRC has the second-highest accuracy and is also the second most time-consuming model, taking 3m 37s (BoW) and 3m 36s (TF-IDF).

• L-SVC stands out as the most eficient model. Despite being marginally less accurate than RFC (0.89 vs. 0.99), it requires significantly less training time-1m 15s for BoW and 1m 14s for TF-IDF-making it 15 times faster than RFC and three times faster than LRC.

• BNB and MNB models are the fastest, with MNB training in just 6 seconds. However, their maximum accuracy of 0.79 is notably lower, making them less suitable for accuracy-critical applications.

• Eficiency Analysis: L-SVC emerges as the most eficient model when considering both accuracy and training time. Although RFC provides the highest accuracy, its extensive training time reduces its eficiency. (Table 6)

Table 9: Classification Reports of All Models Across Dataset Sizes for ML in BoW
<table><tr><td rowspan=1 colspan=12>Classification Reports of All Models Across Dataset Sizes</td></tr><tr><td rowspan=2 colspan=1>Dataset Size</td><td rowspan=2 colspan=1>Model</td><td rowspan=2 colspan=1>Negative</td><td rowspan=2 colspan=2>PrecisionNeutralPositive</td><td rowspan=1 colspan=3>Recall</td><td rowspan=2 colspan=2>F1 ScoreNegativeNeutral</td><td rowspan=2 colspan=1>Positive</td><td rowspan=2 colspan=1>Accuracy</td></tr><tr><td rowspan=1 colspan=1>Positive</td><td rowspan=1 colspan=1>Negative</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>Positive</td><td rowspan=1 colspan=1>Negative</td></tr><tr><td rowspan=5 colspan=1>Imbalanced</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.91</td></tr><tr><td rowspan=4 colspan=1>0.680.410.710.82</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.93</td></tr><tr><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.92</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=5 colspan=1>25% Dataset</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=5 colspan=1>0.840.840.760.860.73</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.89</td></tr><tr><td rowspan=3 colspan=1>0.870.700.89</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.88</td></tr><tr><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.80</td></tr><tr><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.89</td></tr><tr><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.78</td></tr><tr><td rowspan=5 colspan=1>50% Dataset</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.93</td></tr><tr><td rowspan=4 colspan=1>0.860.770.880.75</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=2 colspan=1>0.710.90</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.76</td></tr><tr><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.79</td></tr><tr><td rowspan=5 colspan=1>75% Dataset</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=4 colspan=1>0.900.860.770.88</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.95</td></tr><tr><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.76</td></tr><tr><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.79</td></tr><tr><td rowspan=5 colspan=1>100% Dataset</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.97</td></tr><tr><td rowspan=4 colspan=1>0.870.770.880.74</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.76</td></tr><tr><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.79</td></tr><tr><td rowspan=5 colspan=1>Oversampled</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.99</td></tr><tr><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.76</td></tr><tr><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.79</td></tr></table>

Note: Numerical values are rounded to the nearest decimal point for consistent precision. All distributions are balanced, except for the Imbalanced one.

## 4.7 Comparison of Balanced and Imbalanced Machine Learning Models

When comparing the performance of models trained on imbalanced versus balanced datasets, the diference in prediction quality and fairness becomes evident. In the imbalanced scenario, although models such as RFC, LR, and L-SVC achieved high overall accuracy (91–92%), this performance was misleading. These models were primarily biased toward the dominant positive class, neglecting other classes. For instance, in the BoW approach with the imbalanced dataset as can be seen in Table 7, RFC achieved just 0.5005 precision, 0.4082 recall, and 0.4497 F1-score for the neutral class, revealing the model’s inability to properly classify less-represented classes. This imbalance led to unfair and unreliable predictions, especially in applications where all sentiments must be equally understood.

On the other hand, with a balanced dataset, all three models -RFC, LR, and L-SVCnot only achieved strong accuracy but also demonstrated consistent performance across all sentiment categories. Particularly in the balanced BoW and TF-IDF approaches, RFC delivered near-perfect results (98–99%). Even LRC and L-SVC maintained well-rounded metrics, ofering reliable predictions for every sentiment class,as can be seen in Table 8. The key advantage in the balanced dataset is that models could learn equally from all sentiment categories, resulting in a fairer and more generalizable prediction behavior.

Table 10: Classification Reports of All Models Across Dataset Sizes for ML in TF-IDF Classification Reports of All Models Across Dataset Sizes
<table><tr><td rowspan=2 colspan=1>Dataset Size</td><td rowspan=2 colspan=1>Model</td><td rowspan=2 colspan=1>PrNegative</td><td rowspan=1 colspan=1>ecision</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Recall</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>1 Score</td><td rowspan=2 colspan=1>Positive</td><td rowspan=2 colspan=1>Accuracy</td></tr><tr><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>Positive</td><td rowspan=1 colspan=1>Negative</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1> Positive</td><td rowspan=1 colspan=1>Negative</td><td rowspan=1 colspan=1>Neutral</td></tr><tr><td rowspan=5 colspan=1>Imbalanced</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.86</td></tr><tr><td rowspan=4 colspan=1>0.630.380.450.77</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.88</td></tr><tr><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.79</td></tr><tr><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.86</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=5 colspan=1>25% Dataset</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=4 colspan=1>0.790.710.810.68</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.83</td></tr><tr><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.75</td></tr><tr><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.73</td></tr><tr><td rowspan=5 colspan=1>50% Dataset</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=5 colspan=1>0.840.820.720.830.70</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.88</td></tr><tr><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.74</td></tr><tr><td rowspan=5 colspan=1>75% Dataset</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=5 colspan=1>0.850.820.720.830.71</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=2 colspan=1>0.660.85</td><td rowspan=2 colspan=1>0.770.86</td><td rowspan=2 colspan=1>0.590.79</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.74</td></tr><tr><td rowspan=5 colspan=1>100% Dataset</td><td rowspan=5 colspan=1>RFCLRCBNBL-SVCMNB</td><td rowspan=5 colspan=1>0.910.840.710.830.70</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td></tr><tr><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=2 colspan=1>0.670.87</td><td rowspan=2 colspan=1>0.760.86</td><td rowspan=2 colspan=1>0.590.80</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.75</td></tr><tr><td rowspan=5 colspan=1>Oversampled</td><td rowspan=3 colspan=1>RFCLRCBNB</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.94</td></tr><tr><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=2 colspan=1>L-SVCMNB</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.74</td></tr></table>

Note: Numerical values are rounded to the nearest decimal point for consistent precision. All distributions are balanced, except for the imbalanced one.

Another critical factor is training eficiency. Although RFC consistently had the highest accuracy, it also had the longest training times (approx. 17 min), making it less eficient for practical applications. In contrast, L-SVC ofered a strong balance between speed and accuracy, especially in the balanced dataset, where it completed training in just 1m 14s (TF-IDF) while still achieving 90% accuracy - matching LR’s performance, but in significantly less time.

In summary, using a balanced dataset significantly improves model performance across all sentiment classes. It avoids the misleading results caused by high accuracy in imbalanced datasets and ensures the model learns meaningful patterns for all classes - not just the majority. While RFC remains the top performer in terms of raw accuracy, L-SVC stands out as the most eficient and practical model, especially when considering training time and balanced prediction quality. Therefore, balancing the dataset before training is crucial for building robust sentiment analysis models.

![](images/34c608905268f42bd8b36b36a18006048c9e3268d7efe820c3bc989422114499.jpg)  
Figure 33: Operational Flow Overview for Deep Learning

## 5 Deep Learning Models

In this study, six advanced deep learning models - LSTM, Bi-LSTM, RNN, GRU, Attention and FastText - were employed for sentiment analysis on the Amazon customer reviews dataset. LSTM was selected for its ability to capture long-range dependencies, while Bi-LSTM enhanced this by processing data in both directions. RNN served as a baseline for comparison, despite its limitations in retaining long-term context. GRU was chosen for its computational eficiency with fewer parameters than LSTM. The Attention mechanism allowed the model to focus on important parts of the sequence, improving interpretability and performance. FastText, a shallow neural network, was included for its eficient text representation and robustness to out-of-vocabulary words, making it valuable for large-scale sentiment tasks. These structural diferences ofered a comprehensive evaluation of model performance for sentiment classification. This comprehensive investigation ofers valuable insights into the efectiveness of deep learning architectures for sentiment analysis, emphasizing the impact of model selection and architecture design on classification accuracy and practical deployment scenarios. The operational pipeline depicted in Figure 33 outlines the steps from data collection to decision making.

Sentiment analysis is a crucial Natural Language Processing (NLP) task that aims to determine the sentiment expressed in textual data, categorizing it as positive, negative, or neutral. This task has applications in various domains, including customer feedback analysis, social media monitoring, and market research. Traditional machine learning approaches have been widely used for sentiment analysis, relying on manually crafted features and basic text representations. However, these methods often struggle with the complexities of natural language, such as contextual understanding and handling long-term dependencies.

Deep learning models, by contrast, have demonstrated superior performance in sentiment analysis tasks due to their ability to automatically learn hierarchical features from raw text data. Architectures such as LSTM, RNN, GRU, and Attention-based models capture complex patterns in text, enabling more accurate sentiment predictions. Additionally, embeddings like FastText eficiently represent words with semantic relationships, enhancing model understanding.

This study presents a comprehensive sentiment analysis system for Amazon reviews, leveraging advanced deep learning models to classify sentiments. The dataset contains 568,454 reviews spanning October 1999 to October 2012, with detailed information on products and users. The software system includes an interactive user interface that allows customers to select their preferred deep learning model and input custom reviews for sentiment classification. Performance evaluation is conducted using metrics such as precision, recall, F1-score, accuracy, validation accuracy, loss, and validation loss. Experimentation shows that Attention-enhanced models capture contextual relationships efectively, while FastText handles large-scale data eficiently. This research underscores the transformative potential of deep learning in automating sentiment analysis, delivering accurate insights for informed decision-making.

## 5.1 General Aspects of Deep Learning

## 5.1.1 Preprocessing and Word Embedding

To prepare the Amazon customer review dataset for deep learning-based sentiment analysis, the textual reviews were first cleaned and standardized. Cleaning involved converting all text to lowercase, removing HTML tags, eliminating non-alphabetic characters, and trimming leading or trailing whitespace. Each review score was mapped to three sentiment classes: negative (0), neutral (1), and positive (2). The dataset was then split into training and testing sets using an 80/20 ratio while preserving the original class distribution through stratification.

For the deep learning models, tokenization was performed using Keras’ Tokenizer, limited to the 20,000 most frequent words. The tokenized sequences were standardized to a fixed length of 100 using padding and truncation, ensuring uniform input dimensions for neural network layers such as LSTM, Bi-LSTM, or GRU. Sentiment labels were converted into categorical format suitable for multi-class classification, enabling direct use in embedding and recurrent layers.

Although the same initial cleaning steps were applied for both machine learning and deep learning pipelines, deep learning preprocessing difers primarily in tokenization and input representation. While ML models use vectorization techniques such as BoW and TF-IDF to convert text into fixed-size feature vectors, DL models rely on sequential tokenization followed by embedding layers that preserve word order and semantic relationships. Additionally, DL sequences are padded or truncated to a uniform length to accommodate batch processing, allowing models to capture contextual dependencies and sequential patterns in text, which are not leveraged by traditional ML vector representations.

Word embeddings are a fundamental technique in NLP that represent words as dense vectors in a continuous vector space. An embedding layer with a vocabulary size of 20,000, an embedding dimension of 128, and a sequence length of 100 was used as the first layer in all deep learning models. (Figure 34) This configuration ensures that semantically similar words are positioned closer in the vector space, improving sentiment classification. Unlike traditional methods that rely on sparse and high-dimensional representations such as BoW or TF-IDF, word embeddings capture both semantic and syntactic relationships between words, enabling models to better understand the context and meaning within a sentence.

A primary advantage of word embeddings is that they overcome the limitations of manual feature extraction in NLP tasks. Traditional methods often require handcrafted features and cannot fully capture complex linguistic patterns or contextual meanings. In contrast, embeddings are learned directly from large text corpora, capturing rich semantic information automatically. Moreover, embeddings make future feature extraction unnecessary, as they inherently encode both syntactic and semantic properties of text, serving as complete and compact representations. This simplification of the modeling process improves eficiency, model generalization, and overall accuracy in NLP tasks, including sentiment analysis, text classification, and machine translation. By leveraging word embeddings, this study enhances the capability of deep learning models to accurately and eficiently analyze customer reviews and extract actionable insights.

![](images/be293ce835d9258d5d34aae89b9623c1400c132a5b5626eb3d1e72e07723eb6b.jpg)  
Figure 34: Embedding Coding

## 5.1.2 Training Process

Deep learning models are trained iteratively through multiple epochs, where the dataset is passed through the network repeatedly to refine its parameters and improve classification accuracy. Each epoch adjusts the model based on previous errors, allowing it to learn more efectively over time.

In this sentiment analysis project, the training process was designed to stop if no further increase in accuracy was observed over the last five consecutive iterations. The bestperforming model among all trained epochs was automatically saved to ensure optimal performance. Additionally, the changes between epochs were recorded and will be presented in later sections to compare validation accuracy and loss, providing insights into the training process and model improvements. It is crucial to maintain consistent and balanced changes across epochs, as exemplified in Figure 35, avoiding irregular or erratic fluctuations in performance, whether positive or negative, as such instability may indicate suboptimal learning behavior.

## 5.2 Presentation of Deep Learning Models

Six advanced deep learning models-LSTM, RNN, Bidirectional LSTM, GRU, Attention, and FastText-were implemented to classify sentiment within the dataset. To transform the textual data into meaningful numerical representations suitable for deep learning, word embeddings were applied. These embeddings capture semantic relationships between words, enabling the models to better understand and process the contextual information present in the reviews.

![](images/0d69df69bab77ea1a5646a6aaee104420704d5ea2e1071efd48ec7ee06db2f7d.jpg)

![](images/604bb42a8050b850c3cf4443c8a58b26655543eb5dfd8f4afceed575409b4f8c.jpg)  
Figure 35: Accuracy and Loss Changes Between Epochs

Performance evaluation was conducted using a comprehensive set of metrics, including precision, recall, F1 score, accuracy, validation accuracy, loss, and validation loss. These metrics provided a detailed assessment of the models’ ability to classify sentiments efectively across diferent sentiment categories. The experimental results revealed that Attentionbased models outperformed other architectures by capturing long-range dependencies and contextual relationships within the text, while FastText demonstrated superior eficiency when handling large datasets. The best-performing model is then highlighted, and its results are analyzed to provide insights into its strengths and limitations. Additionally, a detailed comparison between the models is presented to ofer a broader perspective on their influence on sentiment classification performance, highlighting the critical role of architecture design and word embeddings in achieving optimal results.

## 5.2.1 Long Short-Term Memory (LSTM)

The LSTM is a recurrent neural network architecture designed to overcome the limitations of traditional RNNs by capturing long-term dependencies in sequential data. Its unique structure, featuring memory cells and gating mechanisms, allows it to selectively retain or forget information, making it highly efective for processing complex textual data with varying contexts. This capability makes LSTM well-suited for sentiment analysis tasks where understanding the relationship between words over extended sequences is crucial.

In this study, the LSTM model emerged as a highly accurate deep learning solution for sentiment classification using word embeddings on the Amazon customer review dataset. The model achieved an impressive overall accuracy of 98%. Detailed performance metrics indicate precision, recall, and F1-scores of 0.98 to 0.99 for negative, neutral, and positive sentiment classes. Analysis of the confusion matrix reveals that the model correctly classified 70,688 negative, 70,755 neutral, and 69,274 positive instances. Misclassifications were minimal, with 1,124 positive samples incorrectly identified as negative and 967 as neutral, as shown in Figure 36. The support parameter highlights the equal distribution of sentiment classes, with each category containing 71,365 samples. These results underscore the model’s capability to efectively capture and classify sentiment nuances in textual data, demonstrating its robustness and reliability for large-scale sentiment analysis tasks.

Table 11: Architectures and training settings of the deep learning models. All models share the embedding layer, output layer, and training settings listed at the bottom.
<table><tr><td>Model</td><td>Sequence layer</td><td>Pooling / attention</td><td>Dense head</td></tr><tr><td>LSTM</td><td>LSTM, 128 units</td><td></td><td>Dense 64, ReLU</td></tr><tr><td>Bi-LSTM</td><td>Bidirectional LSTM, 128 units</td><td></td><td>Dense 64, ReLU</td></tr><tr><td>GRU</td><td>GRU, 128 units</td><td></td><td>Dense 64, ReLU</td></tr><tr><td>RNN Attention</td><td>SimpleRNN, 128 units LSTM, 128 units,</td><td>Self-attention</td><td>Dense 64, ReLU Dense 64, ReLU</td></tr><tr><td></td><td>return_sequences=True (Attention</td><td>layer on [x, x]), GlobalAver- agePooling1D</td><td></td></tr><tr><td>FastText-style</td><td colspan="3">GlobalAveragePooling1D Dense 64, ReLU</td></tr><tr><td>Shared setting</td><td>Value</td><td></td><td></td></tr><tr><td>Embedding</td><td></td><td>Trainable, randomly initialized (no pretrained vectors); input_dim=20,000, output_dim=128</td><td></td></tr><tr><td>Sequence length Tokenizer</td><td>100 (padding and truncation)</td><td>Keras Tokenizer, num_words=20,000</td><td></td></tr><tr><td>Output layer Loss</td><td>Dense 3, softmax Categorical cross-entropy</td><td></td><td></td></tr><tr><td>Optimizer</td><td></td><td>Adam, learning rate 0.001 (Keras default)</td><td></td></tr><tr><td>Batch size / max epochs Early stopping</td><td>32 / 50 monitor=val_accuracy,</td><td></td><td></td></tr><tr><td></td><td></td><td>restore_best_weights=True</td><td>patience=5,</td></tr><tr><td>Dropout</td><td>None</td><td></td><td></td></tr><tr><td>Data split</td><td>80/20, random_state=42, stratified</td><td></td><td></td></tr></table>

## 5.2.2 Attention

The Attention mechanism enhances deep learning models by dynamically focusing on the most relevant parts of a text sequence, improving their ability to capture contextual dependencies. Unlike traditional recurrent architectures, Attention allows the model to weigh important words diferently, leading to more precise sentiment classification. This mechanism is particularly advantageous for sentiment analysis, where the meaning of a phrase often depends on specific words or contextual relationships within a sentence.

In this study, the Attention-based model demonstrated exceptional performance in sentiment classification using word embeddings on the Amazon customer review dataset. The model achieved an impressive 99% overall accuracy, significantly improving classification robustness. Performance metrics indicate precision, recall, and F1-scores of 0.98 to 0.99 across negative, neutral, and positive sentiment classes. The confusion matrix reveals that the model accurately classified 70,721 negative, 70,921 neutral, and 69,680 positive samples. Misclassifications were minimal, with 895 positive samples misclassified as negative and 790 as neutral, as shown in Figure 37. The balanced class distribution, with each sentiment category containing approximately 71,365 samples, further highlights the model’s consistency. These results underscore the Attention mechanism’s efectiveness in capturing nuanced sentiment variations, making it a powerful tool for large-scale sentiment analysis.

![](images/554095ea43abb98243e69f4dfd96ae036e5ee07c5fd211cbb3b1ac5bc6864f89.jpg)  
Figure 36: LSTM - Confusion Matrix

![](images/1e634ea03e5b043a264c8f6bc86c5025a62f43cdb426fb45c06b91428e87434e.jpg)  
Figure 37: Attention - Confusion Matrix

## 5.2.3 Bidirectional Long Short-Term Memory (Bi-LSTM)

The Bi-LSTM model extends the capabilities of traditional LSTMs by processing input sequences in both forward and backward directions. This bidirectional structure enables the model to capture dependencies from both past and future contexts, making it particularly efective for sentiment analysis, where the sentiment of a word can depend on preceding and succeeding words. By leveraging this dual perspective, the Bi-LSTM model enhances its ability to understand complex linguistic patterns and improves classification accuracy.

In this study, the Bi-LSTM model achieved an overall accuracy of 99% in sentiment classification using word embeddings on the Amazon customer review dataset. The model demonstrated strong performance, with precision, recall, and F1-scores ranging from 0.97 to 0.99 across negative, neutral, and positive sentiment classes. The confusion matrix shows that the model correctly classified 70,662 negative, 70,689 neutral, and 69,486 positive instances. Misclassifications were minimal, with 924 positive samples incorrectly labeled as negative and 955 as neutral, as shown in Figure 38. The dataset maintained a balanced sentiment distribution, with each class containing approximately 71,365 samples. These results highlight the Bi-LSTM model’s ability to efectively capture sentiment nuances by incorporating both past and future context, making it a robust solution for large-scale sentiment analysis.

![](images/c6bee82c627503abeb6edc88b987d835f97b73a7672775c51711c806cc544f0e.jpg)  
Figure 38: Bi-LSTM - Confusion Matrix

## 5.2.4 FastText

FastText is an eficient and lightweight text classification model developed by Facebook AI, designed to capture subword information through its use of character n-grams. Unlike traditional embedding approaches that learn a single vector for each word, FastText represents words as the sum of their subword embeddings, enabling it to handle rare words and spelling variations more efectively. This structure makes FastText computationally eficient while still maintaining strong performance on large-scale NLP tasks.

In this study, the FastText model achieved an overall accuracy of 95% in sentiment classification using word embeddings on the Amazon customer review dataset. Performance metrics indicate precision, recall, and F1-scores of 0.77, 0.80, and 0.97 for the negative, neutral, and positive classes, respectively. The model demonstrated particularly strong performance in identifying positive sentiments, achieving an F1-score of 0.98. However, performance on the negative class was comparatively weaker, with a recall of 0.66 and an F1-score of 0.71, suggesting dificulties in distinguishing subtle negative sentiment cues. Neutral sentiment detection remained moderately strong, with a recall of 0.85 and an F1-score of 0.82, as shown in Figure 39.

Despite its limitations in capturing complex linguistic dependencies compared to recurrent architectures, FastText’s rapid training time of only 3 minutes and 26 seconds highlights its eficiency. These results demonstrate that FastText provides a strong balance between speed and accuracy, making it an efective baseline model for large-scale sentiment classification tasks where computational eficiency is prioritized. However, when compared to more advanced architectures such as LSTM, Bi-LSTM, and Attention, FastText falls short in predictive accuracy due to its inability to model long-range contextual dependencies. Although it is the fastest model in the study, its overall accuracy ranks only above the RNN, indicating that its eficiency comes at the cost of reduced classification performance.

![](images/cb9e60ebe0e6b58a283fb1fbb9e7976105abdc5f9064f780c167d4cbca816c7f.jpg)  
Figure 39: FastText - Confusion Matrix

## 5.2.5 Gated Recurrent Unit (GRU)

The GRU is a streamlined variant of the LSTM architecture, designed to capture longterm dependencies while using fewer parameters and gating mechanisms. GRUs integrate the forget and input gates into a single update gate, allowing for faster training and reduced computational complexity without sacrificing the ability to model sequential patterns. This makes GRUs particularly suitable for sentiment analysis tasks involving long customer reviews.

In this study, the GRU model achieved an overall accuracy of 98% in sentiment classification using word embeddings on the Amazon customer review dataset. The model delivered strong performance across all sentiment categories, with F1-scores of 0.86 for negative, 0.90 for neutral, and 0.99 for positive sentiments. Precision scores reached 0.89, 0.96, and 0.98 for negative, neutral, and positive classes, respectively, while recall scores remained consistently strong, particularly for positive sentiments, where the model achieved a recall of 0.99, as shown in Figure 40.

With a training time of 9 minutes and 10 seconds, the GRU proved to be both eficient and highly accurate. The model’s performance closely aligns with that of LSTM and Bi-LSTM architectures, demonstrating its efectiveness in capturing contextual dependencies in text despite having a simpler structure. These results highlight the GRU as a powerful and computationally eficient alternative for large-scale sentiment analysis. Nevertheless, the GRU did not reach the same level of accuracy achieved by Bi-LSTM or Attentionbased models, which benefit from bidirectional processing and enhanced context modeling. Furthermore, although GRU trains faster than LSTM variants, it is still notably slower than

FastText, placing it in a middle ground where it is neither the most accurate nor the most computationally eficient among the evaluated architectures.

![](images/3993ca6a45302aec8d1506c09c97bf743fe656af45c03bba530800420cac21b4.jpg)  
Figure 40: GRU - Confusion Matrix

## 5.2.6 Recurrent Neural Networks (RNN)

The RNN represents the foundational architecture for sequential text modeling, designed to process input one timestep at a time while maintaining a hidden state. Although RNNs provide a basic mechanism to capture temporal patterns in text, they are limited by vanishing and exploding gradient problems, making it dificult for them to learn long-term dependencies. These limitations often lead to reduced performance compared to more advanced architectures such as LSTM, GRU, and Attention-based models.

In this study, the RNN model achieved an overall accuracy of 92% on the Amazon customer review dataset using word embeddings. The model’s performance varied significantly across the three sentiment classes. While it achieved a high precision of 0.94 and an F1-score of 0.96 for positive sentiments, performance on the negative and neutral classes was substantially weaker. Negative sentiment detection resulted in a recall of 0.29 and an F1-score of 0.38, while the neutral class experienced the lowest performance with a recall of only 0.05 and an F1-score of 0.08, as shown in Figure 41. These results indicate that the RNN struggled to capture complex and nuanced sentiment cues, especially for neutral reviews where contextual understanding is more subtle.

With the longest training time among all models-16 minutes and 53 seconds-the RNN demonstrated both computational ineficiency and reduced predictive accuracy. These findings reinforce the known limitations of traditional RNNs in natural language processing and highlight the importance of more advanced architectures for efective sentiment classification. Given its comparatively low accuracy, especially on negative and neutral classes, and its significantly longer training time, the RNN represents the weakest-performing model in this study. Its inability to capture long-term dependencies places it at a clear disadvantage when compared to LSTM, Bi-LSTM, and Attention models, all of which surpassed 98% accuracy. Even FastText, despite its simpler design, delivered faster training and stronger overall performance, underscoring the limited practical utility of traditional RNNs for large-scale sentiment analysis.

![](images/748fb3ee4e30d290e63bcc1c0839b84ba0624f48b467c5142513b8157401ba3c.jpg)  
Figure 41: RNN - Confusion Matrix

## 5.3 Comparison of Top 3 Models

The Attention-based model, Bi-LSTM, and LSTM were the top-performing deep learning architectures in this sentiment analysis study. Each model leveraged word embeddings to process textual data and exhibited high accuracy in classifying sentiments within the Amazon customer review dataset. Their individual strengths and diferences highlight the impact of model architecture on classification performance.

The Attention-based model emerged as the most efective, achieving 99% overall accuracy. Its primary advantage lies in its ability to dynamically focus on the most relevant parts of a text sequence, allowing it to capture long-range dependencies and contextual relationships. This mechanism significantly improved classification robustness, leading to minimal misclassifications. The confusion matrix revealed that the Attention model efectively distinguished between sentiment classes, demonstrating superior precision and recall across negative, neutral, and positive categories.

The Bi-LSTM model also achieved 99% accuracy, processing input sequences in both forward and backward directions. Unlike standard LSTMs, Bi-LSTM considers both past and future contexts, enhancing its ability to model dependencies within a sentence. While its overall performance closely matched the Attention-based model, slight misclassifications in positive and neutral sentiments indicate that contextual understanding was slightly less precise compared to Attention mechanisms. However, Bi-LSTM’s ability to capture bidirectional dependencies made it a strong competitor.

The LSTM model, with an accuracy of 98%, remained a highly reliable approach. Its memory cell and gating mechanisms allowed it to retain long-term dependencies efectively, making it well-suited for sequential data processing. However, compared to Attention and Bi-LSTM, standard LSTM lacked additional mechanisms for focusing on important words or leveraging bidirectional context, resulting in slightly more misclassified samples.

Overall, while all three models exhibited high precision, recall, and F1 scores, the Attentionbased model outperformed the others by better distinguishing between sentiment classes. Bi-LSTM provided improved context modeling, while LSTM served as a strong baseline for sequential text classification. The results emphasize that architecture design plays a crucial role in optimizing sentiment classification, with Attention-based mechanisms proving particularly advantageous for large-scale sentiment analysis.

![](images/0e1f0905d3b2f96933c91909e73a920a1bf2708a6f9662c3f19152a0f828ab4b.jpg)  
Figure 42: Metrics of Top 3 Model

## 5.3.1 Classification Reports

Figure 42, visually compares the performance of the LSTM, Attention, and Bi-LSTM in a multi-class sentiment analysis task (negative, neutral, positive).

![](images/a7c92f523e0b1185c213d97875d50a258fd2831b7b102376138aca1619719ddf.jpg)  
Figure 43: Classification Report Metrics

In Figure 43, the Attention model exhibits superior performance across all metrics, attributed to its ensemble structure, which enables it to efectively capture complex patterns.

Both the LSTM and Bi-LSTM models demonstrate solid performance, though slightly lower compared to Attention. This comparison underscores Attention’s enhanced adaptability for nuanced sentiment classification, positioning it as the most balanced model for this task, while LSTM and Bi-LSTM also show high accuracy.

## 5.3.2 Running Time Comparison

The Attention-based model demonstrated the longest execution time among the deep learning models, taking 112 minutes and 3 seconds for training. Despite this extended runtime, it achieved the highest overall accuracy of 99%. The ability to dynamically focus on the most relevant parts of a text sequence allowed the Attention model to efectively capture long-range dependencies, significantly enhancing its classification robustness. However, the model’s complex architecture, which includes dynamic weighting of sequence elements, contributed to the increased processing time.

In comparison, the Bi-LSTM model, which also achieved 99% accuracy, had a slightly faster training time of 95 minutes and 20 seconds. By processing input sequences in both forward and backward directions, Bi-LSTM efectively captured bidirectional dependencies within sentences. While its performance was very close to the Attention-based model, the slightly faster runtime and its ability to model both past and future contexts provided a valuable alternative for large-scale sentiment classification tasks. The LSTM model, with an accuracy of 98%, was the quickest of the three, requiring only 53 minutes and 51 seconds for training. Although it lacked the advanced mechanisms of Attention and bidirectional context, its eficient sequential data processing made it a reliable and faster alternative for sentiment classification tasks.

While all three models exhibited high accuracy, the trade-of between running time and performance is evident. The Attention-based model, though the most accurate, comes with a higher computational cost, whereas LSTM provides a faster, yet slightly less precise, solution. The Bi-LSTM ofers a balanced approach, capturing both past and future context with a moderate runtime, making it a competitive model for sentiment analysis in practical applications where training time is a critical consideration.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Running Time</td><td rowspan=1 colspan=1>Accuracy</td></tr><tr><td rowspan=1 colspan=1>LSTM</td><td rowspan=1 colspan=1>53m 51s</td><td rowspan=1 colspan=1>0.984</td></tr><tr><td rowspan=1 colspan=1>RNN</td><td rowspan=1 colspan=1>34m 47s</td><td rowspan=1 colspan=1>0.664</td></tr><tr><td rowspan=1 colspan=1>Bi-LSTM</td><td rowspan=1 colspan=1>95m 20s</td><td rowspan=1 colspan=1>0.986</td></tr><tr><td rowspan=1 colspan=1>GRU</td><td rowspan=1 colspan=1>61m 37s</td><td rowspan=1 colspan=1>0.981</td></tr><tr><td rowspan=1 colspan=1>Attention</td><td rowspan=1 colspan=1>112m 3s</td><td rowspan=1 colspan=1>0.987</td></tr><tr><td rowspan=1 colspan=1>FastText</td><td rowspan=1 colspan=1>29m 54s</td><td rowspan=1 colspan=1>0.979</td></tr></table>

Table 12: Running Time and Accuracy Table

In this section, we assess the performance of various deep learning models used for sentiment analysis, considering both their predictive accuracy and computational eficiency. The

eficiency parameter, defined as

$$
{ \mathrm { E f f i c i e n c y } } = { \mathrm { T i m e } } \times { \mathrm { A c c u r a c y } } ,
$$

combines the model’s accuracy with its training time to ofer a balanced evaluation. Accuracy represents how well the model predicts sentiment, while training time measures the computational resources required. By combining these two factors, the eficiency metric helps identify models that perform well while minimizing computational cost. This is particularly valuable in real-world applications where both speed and accuracy are crucial.

• LSTM achieved an accuracy of 0.9842, with precision, recall, and F1-score all above 0.98. However, it requires more training time compared to simpler models, making it less eficient in terms of speed.

• RNN showed the lowest performance in terms of accuracy (0.6645) among the deep learning models. While it has low training time, its accuracy makes it unsuitable for applications where high precision is critical.

• Bi-LSTM outperformed LSTM with an accuracy of 0.9862, as well as better precision, recall, and F1-scores. This bidirectional approach provides a more comprehensive understanding of context, but it requires additional computational resources compared to standard LSTM.

• GRU exhibited an accuracy of 0.9819, with performance metrics close to those of LSTM. It ofers better eficiency, requiring less training time while maintaining competitive accuracy levels.

• Attention emerged as the most efective model, achieving the highest accuracy (0.9871) among all models, along with precision, recall, and F1-scores in the 0.98 range. Despite its superior performance, the training time is higher, making it a trade-of between speed and accuracy.

• FastText is the fastest model, as highlighted in Table 9, with a significantly lower training time. However, its accuracy (0.9797) is lower than other models, which makes it less suitable for accuracy-critical applications but highly eficient for speed-dependent tasks.

Eficiency Analysis: The Attention model stands out as the best-performing model based on accuracy, while also achieving relatively high eficiency. Although it requires more computational resources, its performance justifies the investment. GRU and Bidirectional LSTM are good alternatives, ofering a balance between accuracy and eficiency. RNN and FastText, while fast, sacrifice accuracy and thus may not be suitable for high-stakes sentiment classification tasks, as shown in Table 10.[12]

<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Performance Metric</td></tr><tr><td rowspan=1 colspan=1>Fastest Model</td><td rowspan=1 colspan=1>FastText</td><td rowspan=1 colspan=1>29m 54s</td></tr><tr><td rowspan=1 colspan=1>Most Accurate Model</td><td rowspan=1 colspan=1>Attention</td><td rowspan=1 colspan=1>0.987 Accuracy</td></tr><tr><td rowspan=1 colspan=1>Most Efficient Model</td><td rowspan=1 colspan=1>Bi-LSTM</td><td rowspan=1 colspan=1> $\mathrm { E } = \mathrm { T } \times \mathrm { A }$ </td></tr></table>

Table 13: Summary of Best Performing Models by Category

Table 14: Classification Reports of Trained Deep Learning Models for Evaluation
<table><tr><td colspan="12">Panel A: Deep Learning Models Classification Report</td></tr><tr><td rowspan="2">Model</td><td colspan="3">Precision</td><td colspan="3">Recall</td><td colspan="3">F1 Score</td><td rowspan="2">Accuracy</td></tr><tr><td>Negative</td><td>Neutral</td><td>Positive</td><td>Negative</td><td>Neutral</td><td>Positive</td><td>Negative</td><td>Neutral</td><td>Positive</td></tr><tr><td>Attention</td><td>0.9048</td><td>0.9286</td><td>0.9849</td><td>0.8416</td><td>0.8661</td><td>0.9923</td><td>0.8721</td><td>0.8963</td><td>0.9886</td><td>0.9780</td></tr><tr><td>11m 6s Bi-LSTM</td><td>0.9022</td><td>0.9426</td><td>0.9837</td><td>0.8340</td><td>0.8478</td><td>0.9921</td><td>0.8668</td><td>0.8927</td><td>0.9879</td><td>0.9770</td></tr><tr><td>15m 27s LSTM</td><td>0.8840</td><td>0.9429</td><td>0.9864</td><td>0.8572</td><td>0.8785</td><td>0.9902</td><td>0.8704</td><td>0.9096</td><td>0.9883</td><td>0.9777</td></tr><tr><td>10m 25s GRU</td><td>0.8854</td><td>0.9593</td><td>0.9837</td><td>0.8335</td><td>0.8428</td><td>0.9909</td><td>0.8587</td><td>0.8973</td><td>0.9873</td><td>0.9758</td></tr><tr><td>9m 10s FastText</td><td>0.7677</td><td>0.7977</td><td>0.9704</td><td>0.6629</td><td>0.8494</td><td>0.9813</td><td>0.7115</td><td>0.8227</td><td>0.9758</td><td>0.9533</td></tr><tr><td>3m 26s RNN</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16m 53s</td><td>0.5605</td><td>0.2372</td><td>0.9366</td><td>0.2896</td><td>0.0499</td><td>0.9900</td><td>0.3819</td><td>0.0825</td><td>0.9626</td><td>0.9185</td></tr></table>

## 5.4 Evaluation of Deep Learning Models Under Class Imbalance

Six DL models were trained and evaluated separately on balanced and imbalanced datasets. In the balanced dataset, models leveraged structured class distributions to enhance learning, with performance metrics analyzed across sentiment categories. Conversely, in the imbalanced dataset, models encountered challenges related to class dominance, afecting their ability to generalize across all sentiment classes. However, despite the advantages of a balanced dataset, DL models demonstrated relatively consistent performance across both balanced and imbalanced datasets. This is because DL architectures can mitigate class imbalance efects through advanced training strategies, loss function tuning, and representational learning. Techniques such as class weighting, mini-batch gradient descent, Attention mechanisms, and contextual embeddings help models generalize well, reducing the need for explicit data balancing.

While it is widely understood that DL models do not necessarily require balanced datasets to perform efectively, we trained them under both balanced and imbalanced conditions to ensure consistency in comparison with traditional ML models. After evaluating the results, we confirmed that performance diferences were minimal across the two settings, validating our prior understanding. Therefore, it was deemed unnecessary to present the imbalanced results for every DL model in this study.

The dataset used for model training was heavily imbalanced, consisting of 356,823 positive, 32,312 negative, and 6,012 neutral samples, totaling 395,147 reviews. Nevertheless, all six DL models demonstrated strong performance, as can be seen on Table 11. This outcome reflects DL models’ stronger ability to handle class imbalance, with a detailed analysis available in [12].

Among the models, the Attention mechanism achieved the highest overall results, reaching 97.8% accuracy and consistently strong precision, recall, and F1-scores across all classes, especially improving the detection of minority classes (negative and neutral). Bi-LSTM and LSTM closely followed Attention in terms of performance, each maintaining over 97% accuracy with similarly balanced metrics. GRU also performed well but slightly lagged behind in precision and recall for minority classes compared to Attention and Bi-LSTM. FastText, although faster in training time (3m 26s), showed a moderate drop in precision and recall, particularly for the negative and neutral classes, highlighting its relative simplicity compared to more complex architectures. RNN, on the other hand, significantly underperformed, especially on the neutral class (only 0.0825 F1-score) and negative class, due to its dificulty in handling longer dependencies and class imbalance, despite showing 91.85% accuracy largely driven by its strong positive class performance.(Table 11)

Overall, the results confirm that modern DL architectures like Attention, Bi-LSTM, and LSTM are robust against class imbalance, successfully capturing patterns even from underrepresented classes. However, simpler or older models like RNN and FastText struggle more under such imbalanced conditions, emphasizing the importance of architectural complexity in achieving balanced and reliable sentiment classification.

## 6 Comparison of Deep Learning and Machine Learning models

Table 15: Classification Reports of Best ML (RFC) and Best DL (Attention) Models Across Dataset Sizes
<table><tr><td>Panel A: Best ML (RFC) and Best DL (Attention) Models Classification Report</td><td colspan="10"></td></tr><tr><td rowspan="2">Dataset Size</td><td rowspan="2">Model</td><td colspan="3">Precision</td><td colspan="3">Recall</td><td colspan="3">F1 Score</td><td rowspan="2">Accuracy</td></tr><tr><td>Negative</td><td>Neutral</td><td>Positive</td><td>Negative</td><td>Neutral</td><td>Positive</td><td>Negative</td><td>Neutral</td><td>Positive</td></tr><tr><td>Imbalanced</td><td>RFC</td><td>0.5671</td><td>0.5005</td><td>0.9336</td><td>0.2689</td><td>0.4082</td><td>0.9809</td><td>0.3648</td><td>0.4497</td><td>0.9567</td><td>0.9143</td></tr><tr><td>P/356,823 Neu/32,312 N/6,012 T/395,147</td><td>Attention</td><td>0.8805</td><td>0.9444</td><td>0.9873</td><td>0.8745</td><td>0.8336</td><td>0.9898</td><td>0.8775</td><td>0.8856</td><td>0.9885</td><td>0.9780</td></tr><tr><td>25% Dataset</td><td>RFC</td><td>0.8400</td><td>0.9700</td><td>0.8700</td><td>0.8600</td><td>1.0000</td><td>0.8200</td><td>0.8500</td><td>0.9800</td><td>0.8400</td><td>0.8900</td></tr><tr><td>P/Neu/N/32,928 T/98,784</td><td>Attention</td><td>0.8209</td><td>0.7787</td><td>0.8568</td><td>0.8270</td><td>0.7933</td><td>0.8344</td><td>0.8239</td><td>0.7859</td><td>0.8455</td><td>0.8182</td></tr><tr><td>50% Dataset</td><td>RFC</td><td>0.8800</td><td>0.9900</td><td>0.9200</td><td>0.9200</td><td>1.0000</td><td>0.8700</td><td>0.9000</td><td>0.9900</td><td>0.9000</td><td>0.9300</td></tr><tr><td>P/Neu/N/65,857 T/197,571</td><td>Attention</td><td>0.8889</td><td>0.8566</td><td>0.9042</td><td>0.8906</td><td>0.8768</td><td>0.8811</td><td>0.8898</td><td>0.8666</td><td>0.8925</td><td>0.8829</td></tr><tr><td>75% Dataset</td><td>RFC</td><td>0.9000</td><td>0.9900</td><td>0.9600</td><td>0.9600</td><td>1.0000</td><td>0.8900</td><td>0.9300</td><td>1.0000</td><td>0.9300</td><td>0.9500</td></tr><tr><td>P/Neu/N/98,786 T/296,358</td><td>Attention</td><td>0.9275</td><td>0.8899</td><td>0.9384</td><td>0.9181</td><td>0.9313</td><td>0.9043</td><td>0.9228</td><td>0.9101</td><td>0.9210</td><td>0.9179</td></tr><tr><td>100% Dataset</td><td>RFC</td><td>0.9700</td><td>0.9800</td><td>0.9800</td><td>0.9800</td><td>0.9800</td><td>0.9700</td><td>0.9700</td><td>0.9800</td><td>0.9700</td><td>0.9722</td></tr><tr><td>P/Neu/N/131,715 T/395,145</td><td>Attention</td><td>0.9526</td><td>0.9526</td><td>0.9526</td><td>0.9531</td><td>0.9531</td><td>0.9531</td><td>0.9529</td><td>0.9529</td><td>0.9529</td><td>0.9532</td></tr><tr><td>Oversampled</td><td>RFC</td><td>0.9600</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9600</td><td>0.9800</td><td>1.0000</td><td>0.9800</td><td>0.9900</td></tr><tr><td>P/Neu/N/356,823 T/1,070,469</td><td>Attention</td><td>0.9861</td><td>0.9851</td><td>0.9899</td><td>0.9910</td><td>0.9938</td><td>0.9764</td><td>0.9886</td><td>0.9894</td><td>0.9831</td><td>0.9871</td></tr></table>

## 6.1 Machine Learning vs Deep Learning for Balanced Data

When both ML and DL models were trained on the oversampled, balanced dataset, a significant improvement was observed across all evaluation metrics compared to training on the imbalanced original dataset and Table 12 is showing all parameters in one table. ML models, particularly RFC and L-SVC, demonstrated substantial gains in precision, recall, F1 score, and overall accuracy, achieving results that were much closer to those obtained by the DL models. For instance, the RFC achieved an accuracy of 99%, while L-SVC and LRC maintained accuracies around 90%, reflecting the importance of balanced class distributions for traditional ML algorithms.[13]

Despite the improvement in ML model performance after balancing the data, DL models continued to outperform ML models slightly across most metrics. The Attention-based DL model, for example, achieved a precision of 0.9861 for negative class, 0.9851 for neutral class, and 0.9899 for positive class, along with an overall accuracy of 98.71%. Furthermore, DL models demonstrated more consistent and stable classification across all classes, with particularly high recall and F1 scores, indicating better generalization to minority classes even in the balanced setting.

However, despite the advantages of a balanced dataset, DL models demonstrated relatively consistent performance across both balanced and imbalanced datasets. This is because DL architectures can mitigate class imbalance efects through advanced training strategies, loss function tuning, and representational learning. Techniques such as class weighting, mini-batch gradient descent, Attention mechanisms, and contextual embeddings help models generalize well, reducing the need for explicit data balancing.

Overall, while balancing the dataset significantly enhanced the performance of ML models and narrowed the gap between ML and DL outcomes, DL models still maintained a slight advantage, particularly in handling subtle variations in the data and achieving consistently high performance across all sentiment categories.

## 6.2 Machine Learning vs Deep Learning for Imbalanced Data

DL models consistently outperformed ML models across all metrics on the imbalanced dataset. ML models, being sensitive to class imbalance, showed strong bias toward the majority class, while DL models leveraged their complex architectures to maintain balanced performance.

ML models with BoW achieved 84–93% accuracy but struggled with minority classes. For example, RFC reached 0.93 precision for the positive class but only 0.27 recall for the negative class. BNB and MNB also performed poorly on the neutral class, with MNB failing completely (Table 7-8).

In contrast, DL models trained with TF-IDF feature extraction achieved consistently higher and more balanced scores across all classes. Models such as Attention, Bi-LSTM, and LSTM achieved accuracy values around 97-98%, with high precision, recall, and F1 scores across negative, neutral, and positive classes. The Attention-based model, for instance, reached an overall accuracy of 97.8%, with F1 scores around 0.87, 0.90, and 0.99 for all classes, respectively. DL models also handled the imbalanced data distribution more efectively, showing strong recall values for all classes, unlike ML models that tended to favor the majority (positive) class as can be seen in Table 11-12.

Additionally, it is worth noting that even faster and simpler DL architectures like Fast-Text outperformed most ML models, demonstrating the overall robustness of DL approaches in handling class imbalance challenges.

In conclusion, DL models, particularly Attention, Bi-LSTM, and LSTM, ofered significantly better performance than Machine Learning models on the imbalanced dataset, highlighting their strength in capturing complex patterns and managing class imbalance more efectively.

## 7 Results

This study presents a comprehensive exploration of both machine learning and deep learning approaches for large-scale sentiment analysis on Amazon customer reviews, emphasizing the critical roles of feature extraction, model selection, and data balancing techniques. Through extensive experimentation, the research demonstrates that algorithmic performance varies significantly depending on accuracy requirements, computational constraints, and the complexity of textual representations.

Among the machine learning models evaluated, RFC achieved the highest accuracy (0.99 with BoW), though at the cost of substantial computational time. In contrast, Linear SVC provided a strong balance between speed and performance, completing classification in just over one minute with an accuracy of 0.89. These findings underscore the importance of aligning model choice with real-world resource and latency requirements. Furthermore, the impact of feature extraction methods such as Bag-of-Words and TF-IDF proved highly influential, and oversampling strategies efectively addressed class imbalance-validated additionally through undersampled datasets to prevent misleading performance gains.

Deep learning experiments expanded this analysis by evaluating LSTM, Bi-LSTM, GRU, RNN, Attention, and FastText models. The Attention model delivered the highest accuracy (0.978), demonstrating strong capability in capturing long-range dependencies and contextual features, albeit with increased computational cost. FastText, by contrast, ofered an optimal eficiency–accuracy trade-of, completing tasks in only three minutes while achieving 0.95 accuracy. These outcomes indicate that deep learning models, especially those leveraging Attention mechanisms or eficient embedding structures, provide substantial advantages for context-rich sentiment tasks. Models such as GRU and LSTM additionally ofer practical midpoints between computational burden and predictive performance.

## 7.1 Comparison of Top 3 Models Among the Machine Learning and Deep Learning Models

To assess the overall eficiency and performance of the sentiment analysis system, the three best-performing models-Attention, LSTM, and RFC-were selected for direct comparison based on their accuracy, training time, and model architecture. (Table 13)

## 7.1.1 Attention Model

The Attention model achieved the highest accuracy among all models, reaching 97.8% on the imbalanced dataset. The architecture combines sequence modeling with an Attention mechanism that allows the network to focus on the most relevant parts of the input sequence. This dynamic focus enables better understanding of context, especially in long and complex sentences, contributing to high performance even on underrepresented classes (neg. and neu.). Although training time for the Attention model was relatively high (approx. 12 minutes), its balanced precision, recall, and F1-scores across all classes make it an ideal choice for tasks where accuracy and robustness matter more than speed.

## 7.1.2 LSTM Model

The LSTM model followed closely behind the Attention model, maintaining an accuracy of over 97%. LSTMs are designed to capture long-term dependencies in sequential data using memory cells and gating mechanisms, which makes them efective in understanding sentence structure and meaning over time. LSTM showed slightly lower performance on minority classes compared to the Attention model but remained consistent overall. Its training time was moderately high, around 10 minutes, due to its sequential computation and complexity.

## 7.1.3 Random Forest Classifier (RFC)

Among the ML models, RFC achieved the highest accuracy of 99% when trained on the balanced dataset using the BoW representation. RFC operates by constructing a multitude of decision trees and aggregating their outputs for prediction, which makes it robust to overfitting and well-suited for structured data. However, it required significantly longer training time-16 minutes and 55 seconds-due to its ensemble nature and complexity in evaluating many decision paths. RFC also relied heavily on balanced class distribution to reach high performance, unlike DL models which performed well even on imbalanced data.

While all three models demonstrated strong performance, the Attention model stands out as the most eficient overall. Despite slightly lower accuracy compared to RFC (97.8% vs. 99%), it achieved balanced and stable performance across all sentiment classes, especially under imbalanced data conditions, which is critical in real-world scenarios. Furthermore, it required less training time than RFC, and its architecture is better suited for capturing nuanced contextual information in text. LSTM followed closely in accuracy and eficiency, while RFC, despite high accuracy, was slower and less robust to imbalance. Thus, considering both accuracy and robustness to imbalance, along with training eficiency, the Attention model is the most eficient and practically reliable model among the three.

## 7.2 Application Phase

In the final stage of this study, a user-friendly sentiment analysis application was developed to enable real-time classification of customer reviews using the trained machine learning (RFC) and deep learning (Attention) models. The interface is intentionally designed with simplicity and accessibility in mind. Users enter text into the “Enter Review” field and initiate analysis by clicking the “Detect” or “Submit” button. The software system then processes the input and displays the predicted sentiment-positive, negative, or neutral-along with an associated confidence score, allowing users to easily interpret the model’s output.

<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Performance Metric</td></tr><tr><td rowspan=1 colspan=1>Fastest Model</td><td rowspan=1 colspan=1>MNB (BoW)</td><td rowspan=1 colspan=1>0m 6s</td></tr><tr><td rowspan=1 colspan=1>Most Accurate Model</td><td rowspan=1 colspan=1>RFC (BoW)</td><td rowspan=1 colspan=1>0.99 Accuracy</td></tr><tr><td rowspan=1 colspan=1>Most Efficient Model</td><td rowspan=1 colspan=1>FastText</td><td rowspan=1 colspan=1>E = T × A</td></tr><tr><td rowspan=1 colspan=1>Best Model</td><td rowspan=1 colspan=1>Attention</td><td rowspan=1 colspan=1>Accuracy ∼ Time</td></tr></table>

Table 16: Summary of Best Performing Models by Category

![](images/cfd51fece71ad57fff62cc42457acf24ed4cafbebfede91e8eef7e7cf4a9d4fd.jpg)  
Figure 44: Application Interface

Additional interface features, such as “Clear” for resetting the input and optional reporting functionality (e.g., a “Flag” button), enhance usability and support practical integration into customer feedback analysis workflows, as illustrated in Figure 44. This seamless interaction between the trained models and the interface ensures an eficient and intuitive user experience, making the application a functional tool for real-time sentiment detection across diverse use cases.

Collectively, the results highlight that there is no universal best model for sentiment analysis; rather, model selection must consider the intended operational environment-whether high precision, real-time processing, scalability, or computational eficiency is prioritized. The insights gained from machine learning and deep learning benchmarks contribute to the development of robust, interpretable, and scalable sentiment analysis systems capable of supporting business decision-making processes such as customer experience optimization, product improvement, and trend monitoring.

Future research can build upon this work by integrating transformer-based architectures like BERT, RoBERTa, and GPT, which promise superior contextual understanding.

Additional opportunities include developing hybrid Attention–recurrent models, exploring multilingual and noisy-text scenarios, and adopting advanced embedding strategies such as subword tokenization, higher-dimensional vector spaces, and longer sequence modeling. Enhancing interpretability, particularly for Attention-driven and deep neural architectures, remains a key area for improving trust and transparency in business-critical sentiment analysis applications. Together, these advancements will contribute to more adaptable, accurate, and real-time sentiment-driven insights across diverse industries.

## 8 Discussion

The findings of this study provide a comprehensive perspective on the comparative efectiveness of traditional machine learning and deep learning architectures for large-scale sentiment analysis. By analyzing more than 500,000 Amazon customer food reviews, the results demonstrate how model performance can vary considerably depending on the representation method, data balance, and architectural complexity. Overall, the outcomes reinforce established observations from previous studies while also ofering new insights into the practical trade-ofs between accuracy, computational eficiency, and robustness to class imbalance.

Consistent with the literature, the deep learning models-particularly LSTM, Bi-LSTM, and Attention-exhibited strong capabilities in capturing long-range contextual dependencies, leading to high predictive accuracy. Prior studies have emphasized the superiority of LSTMbased models over classical machine learning approaches due to their gating mechanisms and ability to manage sequential information more efectively. This trend was replicated in our results, where both LSTM and Bi-LSTM achieved performance exceeding 97%, aligning closely with research reporting accuracies of 90–95% for similar tasks using Amazon datasets. The Attention model, which surpassed all other deep learning architectures in this study with 97.8% accuracy, reflects advancements highlighted in recent literature emphasizing the transformative impact of Attention mechanisms and transformer-based systems. These architectures excel by dynamically focusing on the most informative components of textual input, enabling nuanced understanding of sentiment even in complex or lengthy reviews.

The machine learning results also align with prior work suggesting that classical models remain competitive under specific conditions. RFC achieved a remarkably high accuracy of 99% when trained on a balanced dataset with Bag-of-Words representations, consistent with studies reporting robust performance from ensemble techniques. However, the computational overhead of RFC-nearly 17 minutes of training-mirrors broader critiques that traditional models become ineficient with large feature spaces. Likewise, the strong showing of L-SVC in terms of speed and stability corresponds with earlier research asserting Support Vector Machine as a reliable choice when computational eficiency is prioritized. The dependency of machine learning models on balanced datasets, however, also reinforces findings that these models are notably less resilient to class imbalance than their deep learning counterparts.

The comparative analysis between machine learning and deep learning approaches highlights important implications for real-world sentiment analysis. While deep learning models generally provide superior contextual understanding and robustness, their computational demands may limit deployment in resource-constrained environments. Conversely, machine learning models such as L-SVC, MNB or BNB can ofer fast, interpretable, and lightweight alternatives when accuracy is not the sole priority. The results also illustrate that no model performs universally best across all evaluation dimensions-confirming observations in prior research that model selection must consider domain-specific constraints such as latency requirements, hardware availability, dataset size, and class imbalance.

A notable finding is the performance distribution among deep learning models. Although GRU delivered high accuracy while using fewer parameters than LSTM or Bi-LSTM, it did not exceed the performance of these architectures or the Attention model, aligning with the literature that positions GRU as a computationally eficient but slightly less expressive alternative. FastText demonstrated excellent training speed and reasonable accuracy, confirming claims that subword-based embeddings are particularly efective for large-scale applications; however, its lower performance compared to LSTM-based architectures highlights the limitations of shallow models in capturing deeper semantic cues. The RNN, as expected, delivered the weakest performance, supporting long-standing evidence regarding vanishing-gradient limitations and reinforcing why modern sentiment analysis workflows rarely rely on basic RNN architectures.

The implications of these findings extend beyond model benchmarking. From a practical standpoint, the demonstrated variance in performance and speed underscores the necessity of aligning methodological choices with operational objectives. For organizations prioritizing real-time sentiment monitoring, FastText or L-SVC may be optimal. For applications requiring high precision on nuanced or imbalanced datasets-such as customer satisfaction analytics-Attention-based or LSTM-based architectures represent superior solutions. The successful development and integration of a user-facing sentiment analysis application further demonstrate the feasibility of deploying both machine learning and deep learning models in interactive, real-world settings.

Future research directions are abundant. Incorporating transformer-based models such as BERT, RoBERTa, DistilBERT, or GPT could provide substantial gains in contextual understanding and classification accuracy. Extending the system to multilingual sentiment analysis, noisy user-generated content, or domain adaptation scenarios would improve its applicability across industries. Hybrid approaches combining classical machine learning with deep learning-such as transformer embeddings paired with lightweight classifiers-represent another promising avenue. Moreover, improved interpretability techniques, particularly for Attention-based and recurrent architectures, will be critical for increasing transparency and trust in decision-making environments. Finally, exploring more advanced tokenization and embedding strategies, such as byte-pair encoding or high-dimensional contextual embeddings, may further enhance classification performance.

In summary, this study contributes to the ongoing discourse on sentiment analysis by providing a large-scale, detailed comparison of machine learning and deep learning approaches. The results verify insights from previous works while ofering practical recommendations for selecting appropriate models under various operational constraints. As sentiment analysis continues to play a central role in understanding user behavior and guiding business strategy, the integration of eficient, context-aware, and interpretable modeling frameworks will remain a priority for future research and application development.

## 9 Conclusions

This study provides a comprehensive evaluation of both machine learning and deep learning techniques for large-scale sentiment analysis on Amazon customer reviews. Through systematic experimentation, we demonstrated that model performance is influenced by feature extraction methods, dataset balancing, and architectural complexity. Deep learning models, particularly LSTM, Bi-LSTM, and Attention-based architectures, achieved the highest accuracy and robustness, efectively capturing contextual dependencies in textual data. Among machine learning models, RFC ofered strong predictive performance, though with higher computational cost, while L-SVC presented an eficient alternative when processing speed is critical.

The comparative analysis revealed important trade-ofs between speed, accuracy, and interpretability. FastText and GRU provided eficient solutions with competitive performance, yet they were outperformed by Attention-driven and recurrent architectures in nuanced sentiment detection. Basic RNNs, despite being foundational, showed limited capacity to handle long-term dependencies, resulting in lower accuracy. These findings underscore the necessity of aligning model selection with practical application requirements, considering both computational resources and the complexity of the input data.

The outcomes of this research have practical implications for real-world sentiment analysis software systems, including customer feedback monitoring, product evaluation, and business intelligence applications. Furthermore, the development of a user-oriented sentiment analysis interface demonstrates the feasibility of deploying both high-performing deep learning and traditional machine learning models in interactive settings.

Future work may explore transformer-based architectures, hybrid modeling approaches, multilingual applications, and advanced embedding techniques to further enhance sentiment classification performance. Additionally, improving model interpretability remains a critical area to ensure transparency and trust in sentiment-driven decision-making processes.

In conclusion, this study confirms that while deep learning architectures provide superior contextual understanding and accuracy, careful consideration of computational eficiency and operational constraints is essential. By providing a detailed benchmark across multiple models and feature extraction strategies, this research contributes to the development of robust, scalable, and practically applicable sentiment analysis systems.

## References

[1] J. J. McAuley and J. Leskovec, “From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews” in Proceedings of the 22nd International Conference on World Wide Web (WWW ’13), Rio de Janeiro, Brazil, May 13-17, 2013, pp. 897–908. doi: 10.1145/2488388.2488466

[2] E. Coyne, J. Smit, and L. G¨uner, “Sentiment analysis for Amazon.com reviews” ResearchGate, 2019. doi: 10.13140/RG.2.2.13939.37920.

[3] A. S. M. AlQahtani, “Product sentiment analysis for Amazon reviews” International Journal of Computer Science & Information Technology (IJCSIT), vol. 13, no. 3, pp. 1–10, Jun. 2021.

[4] S. Wassan, X. Chen, T. Shen, M. Waqar, and N. Jhanjhi, “Amazon product sentiment analysis using machine learning techniques” Int. J. Comput. Sci. Netw. Security, vol. 30, pp. 695–703, 2021. doi: 10.24205/03276716.2020.2065.

[5] S. J. Kumaran, “Sentiment analysis of Amazon user reviews using a hybrid approach” in Measurement: Sensors, vol. 27, 2023, Art. no. 100790.

[6] M. Qorich and R. El Ouazzani, “Text sentiment classification of Amazon reviews using word embeddings and convolutional neural networks” in J. of Supercomputing, vol. 79, pp. 11029–11054, 2023.

[7] T. Haque, N. Saber, and F. Shah, “Sentiment analysis on large scale Amazon product reviews” in Proc. of 2018 IEEE Int. Conf. on Innovations in Research and Development (ICIRD’18), Bangkok, Thailand, May 11-12, 2018, pp. 1–6.

[8] N. S. Suryawanshi, “Sentiment analysis with machine learning and deep learning: A survey of techniques and applications,” International Journal of Science and Research Archive, 2024.

[9] L. Ashbaugh and Y. Zhang, “A Comparative Study of Sentiment Analysis on Customer Reviews Using Machine Learning and Deep Learning,” Computers, 2024.

[10] Y. Mamani-Coaquira and E. Villanueva, “A Review on Text Sentiment Analysis With Machine Learning and Deep Learning Techniques,” IEEE Access, vol. 12, pp. 193115–193130, 2024.

[11] M. I. Corak, C. Wang, and X. Zhang, “Sentiment Analysis of Amazon Customer Review by Using Machine Learning Models” in Proc. of the 5th IEEE Int. Conf. on Computer Communication and Artificial Intelligence (CCAI ’25), Haikou, China, May 23-25, 2025, pp. 1–6.

[12] M. I. Corak, C. Wang and X. Zhang, “Sentiment Analysis of Amazon Customer Review by Using Deep Learning Models,” 2025 IEEE 5th International Conference on Software Engineering and Artificial Intelligence (SEAI ’25), Fuzhou, China, 2025, pp. 161-166.

[13] M. I. Corak, C. Wang and X. Zhang, “A Comparative Study of Sentiment Analysis on Amazon Customer Reviews: Deep Learning vs. Machine Learning with Balanced and Imbalanced Datasets,” 2025 IEEE 16th Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON ’25), New York, USA, 2025

[14] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), vol. 1, Minneapolis, USA, 2019, pp. 4171–4186. doi: 10.18653/v1/N19-1423

[15] J. Demˇsar, “Statistical comparisons of classifiers over multiple data sets,” Journal of Machine Learning Research, vol. 7, pp. 1–30, 2006.

[16] S. Garc´ıa and F. Herrera, “An extension on ‘Statistical comparisons of classifiers over multiple data sets’ for all pairwise comparisons,” Journal of Machine Learning Research, vol. 9, pp. 2677–2694, 2008.

[17] T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” Neural Computation, vol. 10, no. 7, pp. 1895–1924, 1998. doi: 10.1162/089976698300017197

[18] B. Pang and L. Lee, “Opinion mining and sentiment analysis,” Foundations and Trends in Information Retrieval, vol. 2, no. 1–2, pp. 1–135, 2008.

[19] A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts, “Learning word vectors for sentiment analysis” in Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, Portland, USA, 2011, pp. 142–150.

[20] A. Vaswani et al., “Attention is all you need” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.

[21] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[22] K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), Doha, Qatar, 2014, pp. 1724–1734.

[23] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter,” arXiv preprint, arXiv:1910.01108, 2019.

[24] Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv preprint, arXiv:1907.11692, 2019.

[25] A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov, “Bag of tricks for eficient text classification” in Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL), Valencia, Spain, 2017, pp. 427–431.

[26] P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, “Enriching word vectors with subword information,” Transactions of the Association for Computational Linguistics, vol. 5, pp. 135–146, 2017.

[27] Y. Kim, “Convolutional neural networks for sentence classification” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), Doha, Qatar, 2014, pp. 1746–1751.

[28] X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks for text classification” in Advances in Neural Information Processing Systems, vol. 28, 2015.

[29] C. J. Hutto and E. Gilbert, “VADER: A parsimonious rule-based model for sentiment analysis of social media text” in Proceedings of the 8th International AAAI Conference on Weblogs and Social Media (ICWSM), Ann Arbor, USA, 2014.

[30] M. Taboada, J. Brooke, M. Tofiloski, K. Voll, and M. Stede, “Lexicon-based methods for sentiment analysis,” Computational Linguistics, vol. 37, no. 2, pp. 267–307, 2011.

[31] S. Kiritchenko, X. Zhu, and S. M. Mohammad, “Sentiment analysis of short informal texts,” Journal of Artificial Intelligence Research, vol. 50, pp. 723–762, 2014.

[32] H. He and E. A. Garcia, “Learning from imbalanced data,” IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263–1284, 2009.

[33] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[34] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.