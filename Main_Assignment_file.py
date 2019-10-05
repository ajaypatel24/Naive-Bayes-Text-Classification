import numpy as np
from numpy import genfromtxt
import pandas as pd
#a = genfromtxt('reddit_test.csv', delimiter=',')
#b = genfromtxt('reddit_test.csv', delimiter=',')
reddit_test = pd.read_csv('reddit_test.csv', sep=',',header=None)
reddit_train = pd.read_csv('reddit_train.csv', sep=',',header=None)
word_list=list()
for i  in comment:
    word_row=i.split(" ")
    for j in word_row:
        word_list.append(j)

comment_id=reddit_train.iloc[1:,0].values.astype(int)
comment=reddit_train.iloc[1:,1].values.astype(str)
subreddit=reddit_train.iloc[1:,2].values.astype(str)
