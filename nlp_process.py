import pandas as pd
import re
from nltk.stem import PorterStemmer
porter=PorterStemmer()


def set_stopwords():
	sw = pd.read_csv("stopwords/english.txt",names=["word"])
	sw_dict = dict.fromkeys(sw["word"], 0)
	# sw_dict = {}
	# for index, row in sw.iterrows():
	# 	# print(row["word"])
	# 	sw_dict[row["word"]]=0
	return sw_dict


def word_extraction(sentence):

    sw_dict = set_stopwords()
    words = re.sub("[^a-zA-Z]", " ",  sentence).split()
    clean_words= []
    for w in words:
    	ws = porter.stem(w.lower())
    	if(len(ws) > 2):
    		clean_words.append(ws)

    cleaned_text = [w for w in clean_words if w not in sw_dict]
    return cleaned_text

def tokenize(allsentences):
    words = []
    for sentence in allsentences:
        w = word_extraction(sentence)
        words.extend(w)
        
    words = sorted(list(set(words)))
    return words

def generate_vocab(allsentences):    
    vocab = tokenize(allsentences)
    vocab = dict(zip(vocab,range(len(vocab))))
    return vocab

