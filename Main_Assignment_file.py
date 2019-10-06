import numpy as np
from numpy import genfromtxt
import pandas as pd
from collections import Counter
import re 

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
what_comment_is_from=list()
counter=0
for i in comment:
    word_row=i.split(" ")
    for j in word_row:
        word_list.append(j)
        what_comment_is_from.append(counter) 
    counter+=1
#removes non alphanumeric character from the strings


print(len(comment))
#print(len(comment) != len(set(comment))) #if true there are duplicates in the list
print(len(subreddit))
'''
for i in range(len(word_list)):
    word_list[i] = re.sub('[^0-9a-zA-Z]+', '', word_list[i])
    
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
    print(i[0], " ", i[1])

transdict = {}
for y in d:
    transdict[y] = ''

#print(transdict)

#print(string.maketrans(transdict))
def removeFromList(the_list, val):
   return [value for value in the_list if value != val]

print(len(word_list))

for i in high:
    print(i[0])
    word_list = removeFromList(word_list, i[0])


print(len(word_list))
'''
'''
delete 50 most common from word_list() and what_comment_is_this_from
dict to link comment to subreddit
get counter for word for each subreddit
'''

CommentToSubreddit = {}

for x, y in zip(comment_id, subreddit): #instead of using comment, use comment_id since duplicates exist
    CommentToSubreddit[x] = y
    


print(len(CommentToSubreddit)) #currently there are 842 comments in the list which are duplicates of other comments


print(CommentToSubreddit[60000])

#print(len(d))















