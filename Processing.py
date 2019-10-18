import numpy as np
from numpy import genfromtxt
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import csv
import re    
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn import tree

class DataPreprocess: 

    def __init__(self, TrainData, TestData):
        self.TrainDataset = TrainData
        self.TestDataset = TestData
        self.comment = TrainData.iloc[:,1]
        self.subreddit = TrainData.iloc[:,-1]
        self.TestComment = TestData.iloc[:,-1]

       

    def ModelEvaluation(self, Dataset, Output, TestSet, TestOutput, Model):


        if (Model == "LR"):
            model = LogisticRegression().fit(Dataset, Output)
        elif (Model == "MNB"):
            model = MultinomialNB(alpha=0.4).fit(Dataset, Output)
        elif (Model == "SVC"):
            model = LinearSVC(random_state=0, tol=1e-5, fit_intercept=True,
                                loss='squared_hinge').fit(Dataset, Output)
        elif (Model == "DTC"):
            model = tree.DecisionTreeClassifier(random_state=0).fit(Dataset, Output)
        elif (Model == "DTR"):
            model = tree.DecisionTreeRegressor().fit(Dataset, Output)

        predictions = model.predict(TestSet)
        
        counter = 0
        #prediction output of test.csv file
        '''
        with open('output.csv','w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['Id','Category'])
                for x in predictions:
                        row = [str(counter), x]
                        writer.writerow(row)
                        counter += 1
        csvFile.close()
        '''
        #predictions = model.predict(TestSet) #predictions

        with open('predict.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['Id','Prediction'])
            for x in predictions:
                row = [str(counter), x]
                writer.writerow(row)
                counter += 1
        csvFile.close()

        print(Model, ":", (model.score(TestSet, TestOutput) * 100)) #accuracy of predictions


       


class LemmaTokenizer(object): #lemmatizer from nltk
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

        
class Stemmer(object): #porterstemmer from nltk
    def __init__(self):
        self.wnl = PorterStemmer()
    def __call__(self, articles):
        return [self.wnl.stem(t) for t in word_tokenize(articles)]

class preprocess_text(object):
    def __init__(self):
        self.text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
        self.text = re.sub('@[^\s]+','USER', text)
        self.text = text.lower().replace("ё", "е")
        self.text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        self.text = re.sub(' +',' ', text)
    def __call__(self,articles):
        return self.text.strip()

reddit_test = pd.read_csv('reddit_test.csv')#, sep=',',header=None)
reddit_train = pd.read_csv('reddit_train.csv')#, sep=',',header=None)


word_list = list() #will hold all words from the document, will be used to generate stopwords 

comment=reddit_train.iloc[1:,1]
'''
counter = 0
for i in comment:
    word_row=i.split(" ")
    for j in word_row:
        word_list.append(j)
        counter+=1

d = {}

for x in word_list:
    if (len(x) <= 4): 
        if x.lower() not in d.keys():
            d[x.lower()] = 1
        else: 
            d[x.lower()] += 1


k = Counter(d)
high = k.most_common(50) #3 most common words removed
low = k.most_common()[:-1000-1:-1]

words_to_remove = []
least_common = []
for x in high:
    words_to_remove.append(x[0])

for x in low:
   least_common.append(x[0])

print(words_to_remove)
#print(least_common)
'''
words_to_remove = ["upvote", "downvote", "upvoted", "downvoted", "this", "&gt;", "*", "Reddit", "reddit", "DAE", "tl;dr","lol", "^", "karma","the","and", "edit", "gold", "silver", "platinum", ""]
obj = DataPreprocess(reddit_train, reddit_test)
g = reddit_test.iloc[:,-1]
#test_size=0.00007
selector = SelectPercentile(f_classif, percentile=10)



TrainX, TestX, TrainY, TestY = train_test_split(obj.comment, obj.subreddit, test_size=0.15, random_state=1, shuffle=True)
#TrainX, TestX, TrainY, TestY = train_test_split(obj.comment, obj.subreddit, test_size=0.15, random_state=2, shuffle=True)



#maybe dont include the lemmatization since it seems to do more bad
'''tokenizer=LemmaTokenizer(),'''
tfidf = TfidfVectorizer( stop_words=words_to_remove, min_df=1, max_df=0.1, lowercase=True,
use_idf=True, smooth_idf=True, strip_accents='unicode', sublinear_tf=True, analyzer='word') #max_features=49500 max_df=1210
#stop_words=words_to_remove, , lowercase=True,
vectorizer = CountVectorizer()

TfOrCV = "TF"

if (TfOrCV == "TF"): #specify TF for tfidf
    TrainX = tfidf.fit_transform(TrainX)
    TestX = tfidf.transform(TestX)
    RealTest = tfidf.transform(obj.TestComment)
elif (TfOrCV == "CV"): #specifc CV for Count Vectorization
    TrainX = vectorizer.fit_transform(TrainX)
    TestX = vectorizer.transform(TestX)
    RealTest = vectorizer.transform(obj.TestComment)


#obj.ModelEvaluation(TrainX,TrainY,RealTest,TestY, "LR") #Real test set LR
#obj.ModelEvaluation(TrainX,TrainY,TestX,TestY, "LR") #regular testing LR
#obj.ModelEvaluation(TrainX, TrainY,RealTest,TestY, "MNB") #Real test set NB scikit
obj.ModelEvaluation(TrainX,TrainY,TestX,TestY, "MNB") #regular testing NB scikit
#obj.ModelEvaluation(TrainX,TrainY,RealTest,TestY, "SVC") #Real test set NB scikit
#obj.ModelEvaluation(TrainX,TrainY,TestX,TestY, "SVC") #regular testing NB scikit
#obj.ModelEvaluation(TrainX,TrainY,RealTest,TestY, "DTC") #Real test set NB scikit
#obj.ModelEvaluation(TrainX,TrainY,TestX,TestY, "DTC") #regular testing NB scikit
#obj.ModelEvaluation(TrainX,TrainY,RealTest,TestY, "DTR") #Real test set NB scikit
#obj.ModelEvaluation(TrainX,TrainY,TestX,TestY, "DTR") #regular testing NB scikit

#Best Trial so far min_df=2, max_df=0.025 test_size=0.05 55.233% on kaggle

#Best Trial on held out test set 55.32% on a 15% held out test set, no tokenizer, min_df=2, max_df=0.025

#best trial on held out set min_df = 1, max_df = 1210, 56%

#best trial 
#stop_words=words_to_remove, min_df=1, max_df=0.1, lowercase=True,use_idf=True, smooth_idf=True, strip_accents='unicode',  sublinear_tf=True,
#test_size=0.00007 accuracy 57.877% on kaggle


