import re
import os

# Read stop words from the file
def read_stopwords(stopwords):
	stopword =[]
	if(os.path.exists(stopwords)):
		file1 = open(stopwords)
		read = file1.readline()
		while(read):
			stopword.append(read.strip())
			read = file1.readline()
	return stopword

# Remove the non words.
def remove_nonwords(text):
	non_words = re.compile(r"[^a-z ]")
	processed_text = re.sub(non_words, ' ', text)
	return processed_text.strip()

# Function to remove stopwords from the text
def remove_stopwords(text, stopwords):
	words = [word for word in text.split() if word not in stopwords]
	return words


# Function to pre-process the text
def preprocess_text(text, stopwords):
	processed_text = remove_nonwords(text.lower())
	words = remove_stopwords(processed_text, stopwords)
	return words

