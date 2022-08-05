import pandas as pd
import numpy as np
from math import log
from nlp_process import word_extraction,generate_vocab


def generate_tf_mat(allsentences,vocab):
	count_sen = len(allsentences)
	count_voc = len(vocab.keys())
	tf_mat=np.zeros((count_sen,count_voc))
	for sen_idx in range(count_sen):
		words = word_extraction(allsentences[sen_idx])
		for w in words:
			if w in vocab:
				tf_mat[sen_idx][vocab[w]] +=1
	tf_mat = tf_mat / count_voc
	return tf_mat

def generate_idf_vec(allsentences,vocab):
	count_sen = len(allsentences)
	count_voc = len(vocab.keys())
	df_vec=np.zeros(count_voc)
	for sen_idx in range(count_sen):
		words = word_extraction(allsentences[sen_idx])
		words_set = set(words)
		for w in words_set:
			if w in vocab:
				df_vec[vocab[w]] +=1
	idf_vec = np.log10(count_sen / df_vec)
	return idf_vec


def generate_tfidf(allsentences):
	vocab = generate_vocab(allsentences)
	tf_mat = generate_tf_mat(allsentences,vocab)
	idf_vec = generate_idf_vec(allsentences,vocab)
	tfidf_mat = tf_mat * idf_vec
	pd.DataFrame(tfidf_mat,columns=vocab.keys()).to_csv("tfidf.csv",index=False)
	print("Matrix is saved in file")
	return tfidf_mat


def process_input(data_folder= "data/", input_file = "clean_tweets.csv"):
	df = pd.read_csv(data_folder + input_file)
	allsentences = df.iloc[:,0]
	tfidf_mat = generate_tfidf(allsentences)
	return

process_input()
print("Completed!")
