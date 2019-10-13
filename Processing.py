import numpy as np
from numpy import genfromtxt
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import re    
from nltk.stem import WordNetLemmatizer 

from sklearn.linear_model import LogisticRegression


class DataPreprocess: 

    def __init__(self, TrainData, TestData):
        self.TrainDataset = TrainData
        self.TestDataset = TestData
        self.comment = TrainData.iloc[:,1]
        self.subreddit = TrainData.iloc[:,-1]
        self.TestComment = TestData.iloc[:,1]
        self.TrainX = 0
        self.TrainY = 0
        self.TestX = 0
        self.TestY = 0

   

    #comment_id=reddit_train.iloc[1:,0].values.astype(int)

    #comment=reddit_train.iloc[1:,1]

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

        

        print(clf.predict(TestSet))
        print(clf.score(TestSet, TestOutput))

       





reddit_test = pd.read_csv('reddit_test.csv')#, sep=',',header=None)
reddit_train = pd.read_csv('reddit_train.csv')#, sep=',',header=None)

obj = DataPreprocess(reddit_train, reddit_test)

TrainX, TestX, TrainY, TestY = train_test_split(obj.comment, obj.subreddit, test_size=0.2, random_state=4)
print(TrainX.shape)
print(TrainY.shape)

tfidf = TfidfVectorizer(stop_words='english')
x = tfidf.fit_transform(TrainX)
tx = tfidf.transform(TestX)


obj.LogisticRegression(x,TrainY,tx,TestY)



