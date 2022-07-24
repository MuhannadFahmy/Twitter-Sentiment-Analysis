# import basic libraries
import sklearn
import numpy as np
import pandas as pd
# import emot
import re
import nltk
nltk.download('stopwords')
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


from emot.emo_unicode import UNICODE_EMO, EMOTICONS
# !pip install emot

# data
data = pd.read_csv("https://raw.githubusercontent.com/Mehrdad93/Workshop/master/Day%205-lab/train.csv")

# Lower Casing --> creating new column called text_lower
data["text_lower"] = data['tweet'].str.lower()
data.head()

# removing punctuation, creating a new column called 'text_punct'
data['text_punct'] = data['text_lower'].str.replace('[^\w\s]', '')
data.head()

# importing stopwords from nltk library (nltk is an important library)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# function to remove the stopwords
def stopwords_f(text):
  
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# applying the stopwords to 'text_punct' and store into 'text_stop'
data["text_stop"] = data["text_punct"].apply(stopwords_f)
data.head()

# function to remove emoji


def remove_emoji(string):
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                  u"\U00002702-\U000027B0"
                                  u"\U000024C2-\U0001F251""]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# passing the emoji function to 'text_stop'
data['text_stop'] = data['text_stop'].apply(remove_emoji)
data.head()

# function for removing emoticons
def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    
    return emoticon_pattern.sub(r'', text)

# applying remove_emoticons to 'text_stop'
data['text_stop'] = data['text_stop'].apply(remove_emoticons)
data.head()

# extract the labels from the train data
y = data['label'].values

# use 70% for the training and 30% for the test
x_train, x_val, y_train, y_val = train_test_split(data['text_stop'].values, y, 
                                                    stratify=y, 
                                                    random_state=1, 
                                                    test_size=0.3, shuffle=True)
documents = ["This is Import Data's Youtube channel",
             "Data science is my passion and it is fun!",
             "Please subscribe to my channel"]

# initializing the countvectorizer
vectorizer = CountVectorizer()

# tokenize and make the document into a matrix
document_term_matrix = vectorizer.fit_transform(documents)

# check the result
pd.DataFrame(document_term_matrix.toarray(), columns = vectorizer.get_feature_names())

# vectorize tweets for model building
vectorizer = CountVectorizer(binary=True)
# if binary is True, all non zero counts are set to 1. This is useful for /
# discrete probabilistic models that model binary events rather than integer counts.

# learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(list(x_train) + list(x_val))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_val_vec = vectorizer.transform(x_val)

# classify using support vector classifier
svm = svm.SVC(kernel = 'linear', probability = True)

# fit the SVC model based on the given data
prob = svm.fit(x_train_vec, y_train).predict_proba(x_val_vec)

# perform classification and prediction on samples in x_test
y_pred_svm = svm.predict(x_val_vec)


print("Accuracy score for SVC is (in percent): ", accuracy_score(y_val, y_pred_svm) * 100)
print(precision_recall_fscore_support(y_val, y_pred_svm, average='macro'))

data = pd.read_csv('https://gitlab.rcg.sfu.ca/mokhtari/df-datasets/-/raw/master/Desktop/DF%20project/df-workshop/Day%202/Application/Day%202-Data/train_word2vec.csv').sample(10000, random_state=0)
data

STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['question1', 'question2']:
        data[col] = data[col].apply(clean_sentence)
    
    return data

data = clean_dataframe(data)
data.head()

def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['question1', 'question2']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(data)        
corpus[0:5]

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=100)
model.wv['trump']

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

    model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=150)
    model.most_similar('trump')