# imports
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
porter=PorterStemmer()
import pandas as pd

#defining regular expression for preprocessing
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")

REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
ONLY_WORDS = re.compile("[^a-zA-Z]")

#function to load the train and test data sets
def load():
	reviews_train = []
	for line in open('./movie_data/full_train.txt', 'r', encoding = "utf8"):
		reviews_train.append(line.strip())

	reviews_test = []
	for line in open('./movie_data/full_test.txt', 'r', encoding = "utf8"):
	    reviews_test.append(line.strip())

	return(reviews_train,reviews_test)


#function to clean up the cluttered review set
def preprocess_reviews(reviews):
	reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
	reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
	reviews = [ONLY_WORDS.sub(" ", line) for line in reviews]
	stemmed_reviews = []
	for line in reviews:
		l=[]
		for word in line.split():
			w1 = porter.stem(word)
			if(len(w1) > 2):
				l.append(w1)
		stemmed_reviews.append(str(l))
	
	return stemmed_reviews



#function to give example of some positive and negative tokens
def token_example(feature_to_coef):
	print("Example of positive words and it's weightage")
	for best_positive in sorted(
	    feature_to_coef.items(), 
	    key=lambda x: x[1], 
	    reverse=True)[:5]:
	    print (best_positive)
	#     ('excellent', 0.9288812418118644)
	#     ('perfect', 0.7934641227980576)
	#     ('great', 0.675040909917553)
	#     ('amazing', 0.6160398142631545)
	#     ('superb', 0.6063967799425831)
	print("Example of negative words and it's weightage")
	for best_negative in sorted(
	    feature_to_coef.items(), 
	    key=lambda x: x[1])[:5]:
	    print (best_negative)
	#     ('worst', -1.367978497228895)
	#     ('waste', -1.1684451288279047)
	#     ('awful', -1.0277001734353677)
	#     ('poorly', -0.8748317895742782)
	#     ('boring', -0.8587249740682945)

def Regularisation_parameter(X_train, y_train, y_val, X_val):
	z = 0
	final_c =0
	for c in [0.01, 0.05, 0.25, 0.5, 1]:
	    lr = LogisticRegression(C=c)
	    lr.fit(X_train, y_train)
	    print ("Accuracy for C=%s: %s" 
	           % (c, accuracy_score(y_val, lr.predict(X_val))))
	    if(z < accuracy_score(y_val, lr.predict(X_val))):
	    	z = accuracy_score(y_val, lr.predict(X_val))
	    	final_c =c
	#	  Approximate data    
	#     Accuracy for C=0.01: 0.87472
	#     Accuracy for C=0.05: 0.88368
	#     Accuracy for C=0.25: 0.88016
	#     Accuracy for C=0.5: 0.87808
	#     Accuracy for C=1: 0.87648
	return final_c

def main():
	#loading the train and test data sets
	reviews_train,reviews_test = load()

	#preprocessing the given data
	reviews_train_clean = preprocess_reviews(reviews_train)
	reviews_test_clean = preprocess_reviews(reviews_test)


	#vectorization of the reviews
	cv = CountVectorizer(binary=True)
	cv.fit(reviews_train_clean)#generates around 92715 features
	# will give a sparse matrix find a way to make this efficient
	X = cv.transform(reviews_train_clean)
	X_test = cv.transform(reviews_test_clean)
	target = [1 if i < 12500 else 0 for i in range(25000)]


	#splitting the train and test data
	X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

	#choosing the regularisaton parameter (logistic regression) for the greatest accuracy value
	best_c_value = Regularisation_parameter(X_train, y_train, y_val, X_val)

	#training the final logistic model for the best accuracy
	final_model = LogisticRegression(C=best_c_value)
	final_model.fit(X, target)
	print ("Final Accuracy: %s" 
	       % accuracy_score(target, final_model.predict(X_test)))
	# Final Accuracy: 0.88128

	feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
	}

	token_example(feature_to_coef)# example of pos and neg sentiment
	feature_df = pd.DataFrame.from_dict(feature_to_coef,orient="index",columns=["sentiment_value"])
	feature_df.to_csv("./sentiment/word_sentiments.csv")
	return 

if __name__ == '__main__':
	main()
