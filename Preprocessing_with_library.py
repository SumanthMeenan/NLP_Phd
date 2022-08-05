import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = vader.polarity_scores(text)
    print(score['pos'], score['neg'])
    return 

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_sentiment(word,tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return [word,-1,-1,-1]

    # lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    # if not lemma:
    #     return ''

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return [word,-1,-1,-1]

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [word,swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()] 


def generate_ifidf(text_corpora):
    
    # list of text documents
    # text = ["The quick brown fox jumped over the lazy dog.",
    #         "The dog.",
    #         "The fox"]
    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocab
    vectorizer.fit(text_corpora)
    # summarize
    #print(vectorizer.vocabulary_.keys())
    #print(vectorizer.idf_)
    # encode document
    vector = vectorizer.transform([text_corpora[0]])
    # summarize encoded vector
    print(vector.shape)
    #print(vector.toarray())
    return vectorizer
ps = PorterStemmer()



stopword_list = set(stopwords.words('english'))
#r_df = pd.read_csv("data/clean_metoo_tweets.csv")

r_df = pd.read_csv("data/clean_tweets.csv")
#print(r_df.loc[:,"clean_txt"])
tagged_s = []
max_len=0
all_senti_val=[]
all_senti_tuple=[]
miss_count=0
all_word_dict = {}
text_corpora = [s.translate(str.maketrans("","","0123456789")) for s in r_df.loc[:,"clean_txt"].dropna()]
vectorizer = generate_ifidf(text_corpora)
pos_val = nltk.pos_tag(vectorizer.vocabulary_.keys())
senti_val=[ get_sentiment(x,y) for (x,y) in pos_val]
final_vocab = [x for x in senti_val if x[1] != -1 ]
#print(final_vocab)
vocab_dict = {}
for item in final_vocab:
    vocab_dict[item[0]]= item[1:].copy()
print(len(vocab_dict.keys()))
text_corpora_senti = [w for s in text_corpora for w in s.split() if w in vocab_dict ]
print(len(text_corpora_senti))
vectorizer_senti = generate_ifidf(text_corpora_senti)
print(len(vectorizer_senti.vocabulary_))
# for s in r_df.loc[:,"clean_txt"].dropna():
#     s = s.translate(str.maketrans("","","0123456789"))    
#     #print()
#     words_data = word_tokenize(s.lower())
#     #print(words_data)
#     words_data = [x for x in words_data if x not in stopword_list ]
    
#     words_data = [ps.stem(x) for x in words_data]
#     #vader_polarity(words_data[0])
#     pos_val = nltk.pos_tag(words_data)
#     senti_val=[ get_sentiment(x,y) for (x,y) in pos_val]

#     if(max_len < len(words_data)):
#         max_len = len(words_data)
#     #print("tagged",t1)
#     all_senti_val.append(senti_val)
#     #all_senti_tuple.append(senti_tuple)

# print("Max word length is ",max_len)
# print("Sentiments missed are",miss_count)
# print(all_senti_val[0])
# print(len(all_senti_val), r_df.shape)
# r_df["Senti_val"]=all_senti_val
# #r_df["Senti_tuple"] = all_senti_tuple
# r_df.to_csv("data/tweet_tagged.csv",index=False)
print("Completed!")
