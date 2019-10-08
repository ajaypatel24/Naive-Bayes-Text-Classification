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
####new###
'''
goes through the list of the last words in the comments 
if the last word is a word the we are going to delete
we move the index back one and the penultimate word becomes the new last word
if that word is still a word to be deleted we keep moving the index back one till we get one that isnt
i.e if comment is 'the cat is cool' 
what_is_the_last_word ='cool'
index=4
word_to_del=cool
=> what_is_last_word='is'
index='3'
Warning I havent put in an error testor if we delete a whole comment i.e say a comment is 'No'
and we want to delete No it will go onto the prev comment
'''
for word_to_del in high:
    for word,i in zip(what_is_the_last_word,range(len(what_is_the_last_word))):
        if word==word_to_del:#if the last word is one of the words were going to delete change it to be the one before
           what_is_the_last_word[i]=word_list[index[i]-1]
           index[i]=index[i]-1 #index goes down aswell
           
           while what_is_the_last_word[i]==word_to_del:#could potentially delte a whole comment 
               what_is_the_last_word[i]=word_list[index[i]-1]
               index[i]=index[i]-1
###end new##             
def removeFromList(the_list, val): #list operation to remove all occurences of a word very quickly
   return [value for value in the_list if value != val]

def removeFromComment(CommentList, val): #removes corresponding removed word from the array of quote ID's (VERY SLOW, BOTTLENECKING EVERYTHING)
    return [value for value in CommentList if word_list[CommentList.index(value)] != val]

#print("start: ", len(word_list))
#print("start what: ", len(what_comment_is_from))

[x.lower for x in word_list] #lowercase everything
'''
for y in high: #iterates over created top x words
    res_list = list(filter(lambda x: word_list[x] == y[0], range(len(word_list)))) #uses lambdas to keep track of all removed indices to apply them to the what_comment_is_from array
    word_list = removeFromList(word_list, y[0]) 
    for x in sorted(res_list, reverse=True): #the run time of this is ridiculous even with 2 words, need to change it somehow
        del what_comment_is_from[x]
        
'''



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















