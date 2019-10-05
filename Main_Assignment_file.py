import numpy as np
from numpy import genfromtxt
import pandas as pd
from collections import Counter

#a = genfromtxt('reddit_test.csv', delimiter=',')
#b = genfromtxt('reddit_test.csv', delimiter=',')
reddit_test = pd.read_csv('reddit_test.csv', sep=',',header=None)
reddit_train = pd.read_csv('reddit_train.csv', sep=',',header=None)
#separates train data into 3 categories
#id0-69999
comment_id=reddit_train.iloc[1:,0].values.astype(int)
#the comment
comment=reddit_train.iloc[1:,1].values.astype(str)
#which subreddit its form
subreddit=reddit_train.iloc[1:,2].values.astype(str)
word_list=list()
#gets the words from the comments
for i  in comment:
    word_row=i.split(" ")
    for j in word_row:
        word_list.append(j)



d = {}
for x in word_list:
    if (len(x) <= 4): 
        if x.lower() not in d.keys():
            d[x.lower()] = 1
        else: 
            d[x.lower()] += 1

k = Counter(d)
high = k.most_common(50)

for i in high:
    print(i[0], " :", i[1]," ")

