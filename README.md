# Construction and Application of Online News Sentiment Index Based on Deep Learning Text Classification
# 基于深度学习文本分类的网络新闻情感指数编制及应用研究

## Abstract
The Consumer Confidence Index is an indicator of the strength of consumer confidence. Since the first consumer confidence index was introduced in 1946, consumer confidence index surveys have often been conducted by means of telephone surveys or questionnaires. In recent years, the survey process has gradually led to some problems, such as the increase in the rate of refusal, and the proportion of elder interviewees is too large, which has a certain impact on the validity of the index. In addition to strengthen the quality control in the design and implementation of the index survey program, we can make a new interpretation of the problem through the big data mining method.

With the rapid development of Internet technology, the Internet has replaced traditional paper media as the main channel for people to obtain and express opinions. The news reflects the public's emotional life status to varying degrees, and people's emotional state is also affected by network media to some extent. Following this intuitive logic, we attempts to construct a consumer confidence index based on online news texts by mining the emotional tendencies of consumers, thereby avoiding some problems in the traditional consumer confidence index survey, and it is timelier and thriftier. However, because there is no authoritative research to prove the direct connection between online news and consumer psychology and behavior, in order to avoid disputes, we refers to the consumer confidence index based on the online news as “Online News Sentiment Index”, which is not directly related to “consumers”, but can be used to measure the attitudes and opinions of consumers reflected in the news text.

The paper starts from the six dimensions (economic development, employment status, price level, living conditions, housing purchase and investment). From Baidu News, we crawled 68,139 news articles related to consumer confidence of 2009.01 to 2018.06, thus obtaining the original text data of this article. First, 5,000 random stories are randomly sampled for each dimension, artificially labeled with “positive”, “neutral” and “negative”, and words in the text are represented as vectors through the word2vec method, using deep learning algorithm such as such as Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN). The text classification algorithm classifies the remaining news, thereby obtaining news texts with emotional tags. Then take the ratio of the difference between the number of "positive" texts and the number of "negative" texts in a quarter as the quarterly index, and then combine the quarterly index into equal weights and add 100 points to get the quarterly Online News Sentiment Index. Then we compare the Online News Sentiment Index with the macroeconomic indicators and the traditional consumer confidence index to illustrate that the Online News Sentiment Index is highly correlated with traditional consumer confidence index, and is partial preemptive and complementary to some macroeconomic indicators. Finally, the Online News Sentiment Index and its sub-indexes are used as independent variables to predict traditional consumer confidence index by time series regression analysis, dynamic regression analysis, VAR and other multivariate time series analysis methods. The model is from simple to complex, which leads to prediction accuracy growing step by step.


## Main Steps and Codes
### Construction of *Online News Sentiment Index*
1. crawling the news from [Baidu News](https://news.baidu.com/)
- code: [baidu_news_crawler.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/1%20baidu_news_crawler.py)
- description:
    - input: search key words and time range of news
    - output: news information in csv type 
    - some key words of six dimensions

    |就业	|投资	|物价	|生活状况	|经济	|购房
    | ------ | ------ | ------ |------ | ------ | ------ |
    |就业率，失业率，就业形势，就业压力，就业前景，就业满意度，求职压力等 	|市场情绪，投资意愿，投资热情，投资情绪等 |通胀预期，通胀压力，物价涨幅，居民物价，物价走势，物价指数，物价满意度等|居民收入，居民幸福，消费意愿，居民消费支出，居民消费能力，生活满意度，居民生活质量等 |经济形势，宏观经济，目前经济，中国经济前景，宏观经济数据，中国的经济发展态势，宏观经济运行等|楼市成交量，购房压力，购房成本，楼市热度，楼市前景，购房意愿，居民楼市信心，购房支出，房价满意度，房价预期等|


2. preprocessing and conbining the news 
- code: [get_all_df.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/2%20get_all_df.py)
- description:
    - delete duplicated and foreign news
    - conbine all the news into all_df.csv 

3. dividing the training set and testing set
- code: [get_train_test.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/3%20get_train_test.py)
- description:
    - divide the training set and testing set according to the year and keywords
    - artificially label the new in training set with “positive”, “neutral” and “negative”

4. new text preprocess
- code: [text_preprocess.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/4%20text_preprocess.py)
- description:
    - Word Segmentation with jieba
    - remove stop words

5. traditional text classification with VSM
- code: [Vector_Space_Model.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/5%20Vector_Space_Model.py)
- description:
    - text representation: TF-IDF
    - classification model ：logistic、Naïve Bayes、SVM
    - best model ：SVM+TF-IDF, accuracy:77%

6. text classification with deep learning algorithm
- code: [word2vec+SVM_CNN](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/tree/master/6%20word2vec%2BSVM_CNN)
- description:
    - text representation: word2vec pretrained by [Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
    - classification model ：CNN、RNN
    - best model ：CNN+word2vec, accuracy:84%

7.  computing *Online News Sentiment Index*
- code: [get_index.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/tree/master/7%20get_index.py)
- description:
    - label unlabeled news with CNN+word2vec 
    - compute six sub-indexes and *Online News Sentiment Index*

### Application of *Online News Sentiment Index*
8. validation analysis of *Online News Sentiment Index*
- description:
    - compare the sub-indexes with the macroeconomic indicators 
    - compare the *Online News Sentiment Index* with traditional consumer confidence indexes, Consumer Confidence Index(CCI) released by National Bureau of Statistics and China Consumer Confidence Index(CCCI) released by academic institutions

![compare the sub-indexes with the macroeconomic indicators](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/pic/validation%20analysis%201.jpg)

![compare the *Online News Sentiment Index* with CCI and CCCI](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/pic/validation%20analysis%202.jpg)

9. predicting CCCI by *Online News Sentiment Index*
- code: [Time_Series_Analysis.R](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/tree/master/8%20Time_Series_Analysis.R)
- description:
    - use *Online News Sentiment Index* to predict CCCI
    - use six sub-indexes to predict CCCI
    - Time Series Analysis method: co-integration, regression, ARIMAX, VAR, VARX

![use *Online News Sentiment Index* to predict CCCI](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/pic/prediction1.jpg)

![use six sub-indexes to predict CCCI](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/pic/prediction2.jpg)

![Time Series Analysis](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/pic/time%20series%20analysis.jpg)

## Research Results
The study shows that the correlation between the Online News Sentiment Index and the China Consumer Confidence Index (CCCI) is as high as 0.86, and has a certain leading effect. The correlation between the fitted index and CCCI is increased to 0.94. The index shows obvious similarity, preemptiveness or complementarity to relevant economic macro indicators. The above results reflect the effectiveness of the Online News Sentiment Index, indicating that online public opinion imposes a certain impact on consumer confidence, and consumer confidence changes can be reflected in news texts. At the same time, the results also show that the time-consuming and costly questionnaire method can be substituted by mining the emotional tendency of online news in a timely and automatic way through computer programs.

