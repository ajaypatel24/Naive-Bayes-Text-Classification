import numpy as np
import pandas as pd
from collections import defaultdict
import re

class NaiveBayes:

    #takes number of categories (subreddits) as input
    def __init__(self, subreddits):
        self.classes = subreddits

    #creating a dictionary with words and their frequencies in each subreddit
    def word_frequencies(self, comment, label_index):
        for word in comment:
            self.dict[label_index][word]+=1 #storing frequencies for each word in each category in dictionary

    #takes training data (X and y) as well as other hyperparameters 
    # as input. Function trains model by modifying model params
    def fit(self, dataset, labels, smoothing):
        self.examples = dataset
        self.labels = labels
        self.dict = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])

        #construct word frequencies for each subreddit (training)
        for subreddit_index, subreddit in enumerate(self.classes):
            comments_from_subreddit = self.examples[self.labels == subreddit]
            cleaned_examples = pd.DataFrame(data = comments_from_subreddit)
            np.apply_along_axis(self.word_frequencies, 1, cleaned_examples, subreddit_index)

        #pre-calculating important values
        prob_subreddits=np.empty(self.classes.shape[0])
        all_words=[]
        word_counts_in_subreddit=np.empty(self.classes.shape[0])

        for subreddit_index, subreddit in enumerate(self.classes):
            #Calculating probability of each subreddit ( p(c) )
            prob_subreddits[subreddit_index]=np.sum(self.labels == subreddit)/float(self.labels.shape[0]) 
            
            #Calculating total counts of all the words of each class 
            count = list(self.dict[subreddit_index].values())
            word_counts_in_subreddit[subreddit_index] = np.sum(np.array(list(self.dict[subreddit_index].values()))) + 1 # vocab is remaining to be added
            
            #get all words of this category                                
            all_words+=self.dict[subreddit_index].keys()
                                                     
        #calculating vocabulary
        self.vocabulary=np.unique(np.array(all_words))
        self.vocabulary_length=self.vocabulary.shape[0]
                                  
        #computing normalizing terms                                      
        denominators = np.array([word_counts_in_subreddit[subreddit_index] + self.vocabulary_length + 1 for subreddit_index, subreddit in enumerate(self.classes)]) 

        #storing all precomputed values as tuples to be more clean
        self.subreddit_values = [(self.dict[subreddit_index], prob_subreddits[subreddit_index], denominators[subreddit_index]) for subreddit_index, subreddit in enumerate(self.classes)]                               
        self.subreddit_values = np.array(self.subreddit_values) 
    
    #gets probability of comment coming from each subreddit
    def get_comment_probability(self, comment):
        subreddit_likelihoods = np.zeros(self.classes.shape[0])

        for subreddit_index, subreddit in enumerate(self.classes):
            for test_token in comment: #split the test example and get p of each test word

    #takes a set of input points X as input and outputs predictions y hat
    def predict(self, dataset):
        y_hat = [] #array to hold predictions
        for comment in dataset:
            conditional_prob = self.get_comment_probability(comment)
            y_hat.append(self.np.argmax(conditional_prob))
        
        return np.array(y_hat)

    #takes true labels and target labels and outputs accuracy
    def evaluate_acc(labels, predictions):

    #script to run k-fold cross validation
    def cross_validation(k):