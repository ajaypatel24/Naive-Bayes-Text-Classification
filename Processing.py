import numpy as np
from numpy import genfromtxt
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import csv

import re    
from nltk.stem import WordNetLemmatizer 

from sklearn.linear_model import LogisticRegression


class DataPreprocess: 

    def __init__(self, TrainData, TestData):
        self.TrainDataset = TrainData
        self.TestDataset = TestData
        self.comment = TrainData.iloc[:,1]
        self.subreddit = TrainData.iloc[:,-1]
        self.TestComment = TestData.iloc[:,-1]
        self.TrainX = 0
        self.TrainY = 0
        self.TestX = 0
        self.TestY = 0

   

    #comment_id=reddit_train.iloc[1:,0].values.astype(int)

    

    #subreddit=reddit_train.iloc[1:,2].values.astype(str)

    def count_vectorize(self, data):
            """
            Vectorizes a reddit comment matrix using pure count
            """
            tfidf = TfidfVectorizer(stop_words='english')
           
            return tfidf.transform(data)
            #self.TestDataSet = vectorizer.transform(self.TestDataset.iloc[:,1])

            
            
    def count_vectorize2(self, data):
            """
            Vectorizes a reddit comment matrix using pure count
            """
            tfidf = TfidfVectorizer(stop_words='english')
            return tfidf.fit_transform(data)
            #self.TestDataSet = vectorizer.transform(self.TestDataset.iloc[:,1])

            
            
    


    def LogisticRegression(self, Dataset, Output, TestSet, TestOutput):

        clf = LogisticRegression().fit(Dataset, Output)

        
        
        g = clf.predict(TestSet)
        
        h = 0
        ''' testing RealOutput
        with open('output.csv','w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['Id','Category'])
                for x in g:
                        row = [str(h), x]
                        writer.writerow(row)
                        h += 1
        csvFile.close()
        '''
        #print(clf.predict(TestSet))
        print(clf.score(TestSet, TestOutput))

       


def Lemmatize(text):
        lemmatizer = WordNetLemmatizer()


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]



reddit_test = pd.read_csv('reddit_test.csv')#, sep=',',header=None)
reddit_train = pd.read_csv('reddit_train.csv')#, sep=',',header=None)
word_list = list()
comment=reddit_train.iloc[1:,1]
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
high = k.most_common(3)

words_to_remove = []
for x in high:
    words_to_remove.append(x[0])

print(" " in words_to_remove)


obj = DataPreprocess(reddit_train, reddit_test)
g = reddit_test.iloc[:,-1]

TrainX, TestX, TrainY, TestY = train_test_split(obj.comment, obj.subreddit, test_size=0.05, random_state=4)
RealTestX = obj.TestComment


tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=words_to_remove, min_df=3, max_df=0.025)
x = tfidf.fit_transform(TrainX)
tx = tfidf.transform(TestX)
g = tfidf.transform(g)
testx = tfidf.transform(RealTestX)

obj.LogisticRegression(x,TrainY,tx,TestY) #third parameter = g: Realtest or testx
 


