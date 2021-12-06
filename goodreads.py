import nltk
#import sklearn
import string
import pickle	# this is for saving and loading your trained classifiers.
import re

#from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
#from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from pathlib import Path
from string import digits


def preprocess(filename):

	#		TO DO: Apply basic text processing steps which you think are required. (tokenizationi stemming etc.)       #
	#  			   If you want to do further preprocessing (e.g. removing number etc.), you can apply those here.      #

	documentClass = filename.split("_")[0];
	documentType = filename.split("_")[1].split(".txt")[0];
	filepath = Path("data/"+documentType+"/" + filename);
	file = open(filepath, 'r')
	lines = file.read().splitlines()
	file.close()

	processed = [];
	lineCounter = 0;
	bookName = "";

	for line in lines:
		# Combine book name - description together
		if (lineCounter % 2) == 0:
			bookName = line;
		else:

			line = bookName + " " + line;
			line = line.strip(); # Strip all extra white spaces
			line = line.translate({ord(k): None for k in digits}) # Remove all numbers in the string
			# I will turn all letters to lower case since our goal is to classify depending on topic
			line = line.lower();

			# Expand contaracted words.
			line = decontracted(line);

			# remove punctuations
			line = line.translate(str.maketrans('', '', string.punctuation));

			# remove the stop words
			line = remove_stopwords(line);

			# instead of stemming words, I will lemmatize them with wordnet article database
			line = lemmatize_words(line);

			# remove single letter words in the text like: j. f. kennedy => kennedy
			line = ' '.join( [w for w in line.split() if len(w)>1] )

			processed += [(documentClass, line)];

		lineCounter+=1;



	return processed 	# you may change the return value if you need.


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


stop_words = set(stopwords.words('english'))
def remove_stopwords(data):
    temp_list=[]
    for word in data.split():
        if word.lower() not in stop_words:
            temp_list.append(word)
    return ' '.join(temp_list)



lemma = nltk.wordnet.WordNetLemmatizer()
lemma.lemmatize('article')
def lemmatize_words(text):
    return " ".join([lemma.lemmatize(word) for word in text.split()])



######################################################################################################################################################
# This part is not compulsory. However, merging preprocessed forms of your files into mega documents may be pretty helpful.
# You may also want to permanently store (i.e. write to a file) those mega documents so as not to preprocess your file again and again at each trial.
def create_training_megadoc():
	training_documents = ["philosophy_train.txt","sports_train.txt","mystery_train.txt","religion_train.txt","science_train.txt","romance_train.txt","horror_train.txt","science-fiction_train.txt"]
	training_megadoc = []

	for filename in training_documents:
		training_megadoc += preprocess(filename)
	#####
	#...
	# Here, you may write the training_megadoc to a file. (You may also do it elsewhere or nowhere.)
	#...
	#####
	return training_megadoc


def create_test_megadoc():
	training_documents = ["philosophy_test.txt","sports_test.txt","mystery_test.txt","religion_test.txt","science_test.txt","romance_test.txt","horror_test.txt","science-fiction_test.txt"]
	test_megadoc = []
	#...
	#...
	#... *** TO DO ***
	#...
	#...
	return test_megadoc

####################################################################################################################################################






def extract_features(megadoc):
	return		# megadoc can be either training_megadoc for training phase or test_megadoc for testing phase.
	####################################################################################################################
	#																												   #
	#		TO DO: Select features and create feature-based representations of labeled documents.                      #
	#																												   #
	####################################################################################################################






def train(classifier, training_set):	# classifier is either nltk.NaiveBayesClassifier or SklearnClassifier(SVC()). Example call: train(SklearnClassifier(SVC()), training_set)
	return
	####################################################################################################################
	#																												   #
	#		TO DO: Use feature-based representations of your labeled documents to train your classifier.			   #
	#																												   #
	####################################################################################################################






def test(classifier, test_set):
	return
	####################################################################################################################
	#																												   #
	#		TO DO: Use feature-based representations of your labeled documents to test your trained classifier.		   #
	#	 Compute accuracy, recall, precision values and confusion matrices. Present and discuss them at your report.   #
	#																												   #
	####################################################################################################################






def save_classifier(classifier, filename):	#filename should end with .pickle and type(filename)=string
	with open(filename, "wb") as f:
		pickle.dump(classifier, f)
	return


def load_classifier(filename):	#filename should end with .pickle and type(filename)=string
	classifier_file = open(filename, "rb")
	classifier = pickle.load(classifier_file)
	classifier_file.close()
	return classifier




if __name__ == "__main__":
	# You may add or delete global variables.
	training_set = []
	test_set = []

	print(preprocess("philosophy_train.txt"));
