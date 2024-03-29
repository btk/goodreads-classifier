import nltk
import sklearn
import string
import pickle	# this is for saving and loading your trained classifiers.
import re

from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.metrics.scores import (precision, recall)
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from pathlib import Path
from string import digits


def preprocess(filename):
	print("Preprocessing... " + filename);
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

			processed += [(line, documentClass)];

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



def create_training_megadoc():
	training_documents = ["philosophy_train.txt","sports_train.txt","mystery_train.txt","religion_train.txt","science_train.txt","romance_train.txt","horror_train.txt","science-fiction_train.txt"]
	training_megadoc = []

	for filename in training_documents:
		training_megadoc += preprocess(filename)
	return training_megadoc


def create_test_megadoc():
	test_documents = ["philosophy_test.txt","sports_test.txt","mystery_test.txt","religion_test.txt","science_test.txt","romance_test.txt","horror_test.txt","science-fiction_test.txt"]
	test_megadoc = []


	for filename in test_documents:
		test_megadoc += preprocess(filename)

	return test_megadoc


def extract_features(megadoc):
	# bag of words 1-gram
	from nltk.tokenize import word_tokenize
	from itertools import chain

	print("Megadoc length: ", len(megadoc));
	vocabulary = set(chain(*[i[0].split(" ") for i in megadoc]))
	feature_set = [({i:(i in description.split(" ")) for i in vocabulary},tag) for description, tag in megadoc]

	# use word_tokenize

	# vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in megadoc]))
	# feature_set = [({i:(i in word_tokenize(description.lower())) for i in vocabulary},tag) for description, tag in megadoc]

	return feature_set;




def train(classifier, training_set):
	print("Training...");
	return classifier.train(training_set)




def test(classifier, test_set):
	print("Testing...");
	count_right = 0;
	count_wrong = 0;


	for book in test_set:
		classifier_guess = classifier.classify(book[0]);
		if classifier_guess == book[1]:
			count_right += 1;
		else:
			count_wrong += 1;

	accuracy = count_right / (count_wrong + count_right);
	return accuracy;





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

	load_from_pickle = 0

	if load_from_pickle == 0:
		training_megadoc = create_training_megadoc();
		training_set = extract_features(training_megadoc)

	test_megadoc = create_test_megadoc();
	test_set = extract_features(test_megadoc)

	classifier_type = "naivebayes" # naivebayes, svc

	if load_from_pickle == 1:
		print("Loading from pickle...")
		classifier = load_classifier(classifier_type+".pickle")
	else:
		classifier = train(nltk.NaiveBayesClassifier, training_set);
		save_classifier(classifier, classifier_type+".pickle")

	accuracy = test(classifier, test_set);
	print("NaiveBayes Classifier Accuracy: ")
	print(accuracy);

	classifier_type = "svc" # naivebayes, svc

	if load_from_pickle == 1:
		print("Loading from pickle...")
		classifier = load_classifier(classifier_type+".pickle")
	else:
		classifier = train(SklearnClassifier(SVC()), training_set);
		save_classifier(classifier, classifier_type+".pickle")

	accuracy = test(classifier, test_set);
	print("SVC Classifier Accuracy: ")
	print(accuracy);
