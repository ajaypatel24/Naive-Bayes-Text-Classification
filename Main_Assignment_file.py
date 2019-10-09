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
#print(len(comment) != len(set(comment))) #if true there are duplicates in the list, already verified that this is the case
print(len(subreddit))

for i in range(len(word_list)):
    word_list[i] = re.sub('[^0-9a-zA-Z]+', '', word_list[i])
    
d = {}

for x in word_list:
    if (len(x) <= 4): 
        if x.lower() not in d.keys():
            d[x.lower()] = 1
        else: 
            d[x.lower()] += 1


#orders dictionary entries from highest to lowest 
k = Counter(d)
high = k.most_common(2)


def removeFromList(the_list, val): #list operation to remove all occurences of a word very quickly
   return [value for value in the_list if value != val]

def removeFromComment(CommentList, val): #removes corresponding removed word from the array of quote ID's (VERY SLOW, BOTTLENECKING EVERYTHING)
    return [value for value in CommentList if word_list[CommentList.index(value)] != val]

#print("start: ", len(word_list))
#print("start what: ", len(what_comment_is_from))

[x.lower for x in word_list] #lowercase everything

for y in high: #iterates over created top x words
    res_list = list(filter(lambda x: word_list[x] == y[0], range(len(word_list)))) #uses lambdas to keep track of all removed indices to apply them to the what_comment_is_from array
    word_list = removeFromList(word_list, y[0]) 
    for x in sorted(res_list, reverse=True): #the run time of this is ridiculous even with 2 words, need to change it somehow
        del what_comment_is_from[x]
        

#store index of final word in comment 






print("post word: ", len(word_list))
print("post what: ", len(what_comment_is_from))

'''
delete 50 most common from word_list() and what_comment_is_this_from done
dict to link comment to subreddit done
get counter for word for each subreddit (need to do)
'''

CommentToSubreddit = {}

for x, y in zip(comment_id, subreddit): #instead of using comment, use comment_id since duplicates exist
    CommentToSubreddit[x] = y
    


print(len(CommentToSubreddit)) 

#print(len(d))









