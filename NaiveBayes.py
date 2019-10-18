import numpy as np
import pandas as pd
from collections import defaultdict
import re
from Processing import DataPreprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_string(str_arg):
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    
    return cleaned_str # returning the preprocessed string 

class NaiveBayes:

    #takes number of categories (subreddits) as input
    def __init__(self, subreddits):
        self.classes = subreddits

    #creating a dictionary with words and their frequencies in each subreddit
    def word_frequencies(self, comment, label_index):
        if isinstance(comment,np.ndarray): comment=comment[0]
        for word in comment.split():
            self.dict[label_index][word]+=1 #storing frequencies for each word in each category in dictionary

    #takes training data and subreddit names as input, computes frequencies
    def fit(self, dataset, subreddit_names):
        self.examples = dataset
        self.labels = subreddit_names
        self.dict = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])

        #construct word frequencies for each subreddit (training)
        for subreddit_index, subreddit in enumerate(self.classes):
            comments_from_subreddit = self.examples[self.labels == subreddit]
            cleaned_examples=[preprocess_string(comment_subr) for comment_subr in comments_from_subreddit]
            cleaned_examples = pd.DataFrame(data = comments_from_subreddit)
            np.apply_along_axis(self.word_frequencies, 1, cleaned_examples, subreddit_index)

        #pre-calculating important values
        prob_subreddits=np.empty(self.classes.shape[0])
        all_words=[]
        word_counts_in_subreddit=np.empty(self.classes.shape[0])

        for subreddit_index, subreddit in enumerate(self.classes):
            #Calculating probability of each subreddit ( p(c) )
            prob_subreddits[subreddit_index]=np.sum(self.labels == subreddit)/float(self.labels.shape[0]) 
            count = list(self.dict[subreddit_index].values())
            word_counts_in_subreddit[subreddit_index] = np.sum(np.array(count)) + 1 # vocab is remaining to be added                              
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
        subreddit_likelihoods = np.zeros(self.classes.shape[0]) #to store probability w.r.t each class
        for subreddit_index, subreddit in enumerate(self.classes): 
            for word in comment.split(): #split the test example and get p of each test word                                                     
                word_counts = self.subreddit_values[subreddit_index][0].get(word, 0) + 1                          
                word_prob = word_counts/float(self.subreddit_values[subreddit_index][2])                              
                subreddit_likelihoods[subreddit_index] += np.log(word_prob)
                                              
        conditional_prob = np.empty(self.classes.shape[0])
        for subreddit_index, subreddit in enumerate(self.classes):
            conditional_prob[subreddit_index] = subreddit_likelihoods[subreddit_index] + np.log(self.subreddit_values[subreddit_index][1])                                  
      
        return conditional_prob

    #takes a set of input points X as input and outputs predictions y hat
    def predict(self, test_set):
        y_hat = [] #array to hold predictions
        for comment in test_set:
            cleaned_example=preprocess_string(comment)
            conditional_prob = self.get_comment_probability(cleaned_example)
            y_hat.append(self.classes[np.argmax(conditional_prob)])
        
        return np.array(y_hat)

    #takes true labels and target labels and outputs accuracy
    def evaluate_acc(self, test_labels, predictions):
        return np.sum(predictions==test_labels)/float(test_labels.shape[0])

    #script to run k-fold cross validation
    #def cross_validation(k):

###########################################################################

#Starting experiments
reddit_train = pd.read_csv('reddit_train.csv')
reddit_test = pd.read_csv('reddit_test.csv')

preprocess = DataPreprocess(reddit_train, reddit_test)

TrainX, TestX, TrainY, TestY = train_test_split(preprocess.comment, preprocess.subreddit, test_size=0.05, random_state=4)
RealTestX = preprocess.TestComment

n_bayes = NaiveBayes(np.unique(TrainY))
n_bayes.fit(TrainX, TrainY)
predictions = n_bayes.predict(TestX)
accuracy = n_bayes.evaluate_acc(TestY, predictions)

print(accuracy)
