import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
from Processing import DataPreprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_string(str_arg):
    clean_comment = re.sub('[^a-z\s]+',' ',str_arg,flags = re.IGNORECASE) 
    clean_comment = re.sub('(\s+)',' ',clean_comment)
    clean_comment = clean_comment.lower()
    
    return clean_comment

class NaiveBayes:
    
    #takes number of categories (subreddits) as input
    def __init__(self, subreddits):
        self.classes = subreddits
        
    #creating a dictionary with words and their frequencies in each subreddit
    def word_frequencies(self, comment, label_index):
        if isinstance(comment,np.ndarray): comment = comment[0]
        for word in comment.split():
            self.frequency_dict[label_index][word] += 1 #storing frequencies for each word in each category in dictionary
            
    #takes training data and subreddit names as input, computes frequencies
    def fit(self, dataset, subreddit_names):
    
        self.examples = dataset
        self.labels = subreddit_names
        self.frequency_dict = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
            
        #construct word frequencies for each subreddit (training)
        for subreddit_index,subreddit in enumerate(self.classes):
            comments_from_subreddit = self.examples[self.labels == subreddit] #looking at examples from single subreddit at a time
            
            cleaned_comments = [preprocess_string(comment_subr) for comment_subr in comments_from_subreddit]
            cleaned_comments = pd.DataFrame(data = cleaned_comments)
            
            #populating word frequency dictionary
            np.apply_along_axis(self.word_frequencies,1,cleaned_comments,subreddit_index)

        #pre-calculating important values
        subreddit_probabilities = np.empty(self.classes.shape[0])
        all_words_in_dataset = []
        word_counts_in_subreddit = np.empty(self.classes.shape[0])

        for subreddit_index,subreddit in enumerate(self.classes):
            #Calculating prior probability of each subreddit ( p(c) )
            subreddit_probabilities[subreddit_index] = np.sum(self.labels == subreddit) / float(self.labels.shape[0]) 
            #Calculating word counts within the subreddits 
            count = list(self.frequency_dict[subreddit_index].values())
            word_counts_in_subreddit[subreddit_index] = np.sum(np.array(list(self.frequency_dict[subreddit_index].values()))) + 1
            #adding to total word count                               
            all_words_in_dataset += self.frequency_dict[subreddit_index].keys()
                                  
        #computing normalizing terms                              
        denominators = np.array([word_counts_in_subreddit[subreddit_index] + 2 for subreddit_index,subreddit in enumerate(self.classes)])                                                                          
        
        #storing all precomputed values as tuples to be more clean
        self.subreddit_values = [(self.frequency_dict[subreddit_index],subreddit_probabilities[subreddit_index],denominators[subreddit_index]) for subreddit_index,subreddit in enumerate(self.classes)]                               
        self.subreddit_values = np.array(self.subreddit_values)                                 
                                              
    #gets probability of comment coming from each subreddit                                          
    def get_comment_probability(self,comment):
        subreddit_likelihoods = np.zeros(self.classes.shape[0])
        #calculating prior probabilities for each subreddit
        for subreddit_index, subreddit in enumerate(self.classes):            
            for word in comment.split():                            
                word_counts = self.subreddit_values[subreddit_index][0].get(word,0) + 1                           
                word_likelihood = word_counts / float(self.subreddit_values[subreddit_index][2])                              
                subreddit_likelihoods[subreddit_index] += np.log(word_likelihood) #log likelihood
                                              
        posterior_prob = np.empty(self.classes.shape[0])
        for subreddit_index,subreddit in enumerate(self.classes):
            posterior_prob[subreddit_index] = subreddit_likelihoods[subreddit_index]+np.log(self.subreddit_values[subreddit_index][1])                                  
        return posterior_prob
    
    #takes a set of input points X as input and outputs predictions y hat
    def predict(self, X): 
        y_hat = [] #holds predictions
        for comment in X:                                  
            cleaned_comment = preprocess_string(comment)                                  
            posterior_prob = self.get_comment_probability(cleaned_comment) 
            y_hat.append(self.classes[np.argmax(posterior_prob)])
        return np.array(y_hat) 
    
    #takes true labels and target labels and outputs accuracy
    def evaluate_acc(self, test_labels, predictions):
        return np.sum(predictions == test_labels) / float(test_labels.shape[0])

###########################################################################

#Starting experiments
reddit_train = pd.read_csv('reddit_train.csv')
reddit_test = pd.read_csv('reddit_test.csv')

preprocess = DataPreprocess(reddit_train, reddit_test)

TrainX, TestX, TrainY, TestY = train_test_split(preprocess.comment, preprocess.subreddit, test_size = 0.05, random_state = 4)
RealTestX = preprocess.TestComment

n_bayes = NaiveBayes(np.unique(TrainY))
n_bayes.fit(TrainX, TrainY)
predictions = n_bayes.predict(TestX)
accuracy = n_bayes.evaluate_acc(TestY, predictions)

print(accuracy)