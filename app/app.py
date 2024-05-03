import streamlit as st
import skops.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Load Vectorizer
def custom_tokenizer(text, max_words=50):
    tokens = text.split()[:max_words]
    return tokens
unknown_types = sio.get_untrusted_types(file="tfidf_vectorizer.skops")
vectorizer = sio.load("tfidf_vectorizer.skops", trusted=unknown_types)

# Load model
unknown_types = sio.get_untrusted_types(file="logistic_sentiment.skops")
model = sio.load("logistic_sentiment.skops", trusted=unknown_types)

# Download stopwords
pt_stopwords = stopwords.words('portuguese')

# Defining a function to plot the sentiment of a given phrase
def sentiment_analysis(text, pipeline, vectorizer, model):
    
    # Applying the pipeline
    if type(text) is not list:
        text = [text]
    text_prep = pipeline.transform(text)
    matrix = vectorizer.transform(text_prep)
    
    # Predicting sentiment
    pred = model.predict(matrix)
    proba = model.predict_proba(matrix)
    return pred, proba
    
# Class for regular expressions application
class ApplyRegex(BaseEstimator, TransformerMixin):
    
    def __init__(self, regex_transformers):
        self.regex_transformers = regex_transformers
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Applying all regex functions in the regex_transformers dictionary
        for regex_name, regex_function in self.regex_transformers.items():
            X = regex_function(X)
            
        return X
    
def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]

# Class for stopwords removal from the corpus
class StopWordsRemoval(BaseEstimator, TransformerMixin):
    
    def __init__(self, text_stopwords):
        self.text_stopwords = text_stopwords
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [' '.join(stopwords_removal(comment, self.text_stopwords)) for comment in X]

def stemming_process(text, stemmer=RSLPStemmer()):
    return [stemmer.stem(c) for c in text.split()]

# Class for apply the stemming process
class StemmingProcess(BaseEstimator, TransformerMixin):
    
    def __init__(self, stemmer):
        self.stemmer = stemmer
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]
    
# Class for extracting features from corpus
class TextFeatureExtraction(BaseEstimator, TransformerMixin):
    
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return self.vectorizer.fit_transform(X).toarray()

def re_breakline(text_list):
    # Applying regex
    return [re.sub('[\n\r]', ' ', r) for r in text_list]

def re_hiperlinks(text_list):
    # Applying regex
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, ' link ', r) for r in text_list]

def re_dates(text_list):
    # Applying regex
    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, ' data ', r) for r in text_list]

def re_money(text_list):
    # Applying regex
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, ' dinheiro ', r) for r in text_list]

def re_numbers(text_list):    
    # Applying regex
    return [re.sub('[0-9]+', ' numero ', r) for r in text_list]

def re_negation(text_list):
    # Applying regex
    return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', r) for r in text_list]

def re_special_chars(text_list):
    # Applying regex
    return [re.sub('\W', ' ', r) for r in text_list]

def re_whitespaces(text_list):
    # Applying regex
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end

regex_transformers = {
    'break_line': re_breakline,
    'hiperlinks': re_hiperlinks,
    'dates': re_dates,
    'money': re_money,
    'numbers': re_numbers,
    'negation': re_negation,
    'special_chars': re_special_chars,
    'whitespaces': re_whitespaces
}

text_pipeline = Pipeline([
    ('text_features', TextFeatureExtraction(vectorizer))
])

prod_pipeline = Pipeline([
    ('regex', ApplyRegex(regex_transformers)),
    ('stopwords', StopWordsRemoval(stopwords.words('portuguese'))),
    ('stemming', StemmingProcess(RSLPStemmer()))
])
vectorizer = text_pipeline.named_steps['text_features'].vectorizer


# Title of the app
st.title("Review Sentiment Analysis App")


# Text area for user input
user_input = st.text_area("Enter text for sentiment analysis")


# Button to trigger sentiment analysis
if st.button("Analyze"):
    if user_input == "":
        st.error("Please enter some text.")
    else:
        pred, proba = sentiment_analysis(user_input, pipeline=prod_pipeline, vectorizer=vectorizer, model=model)
        fig, ax = plt.subplots(figsize=(5, 3))
        if pred[0] == 1:            
            text = 'Positive'
            class_proba = 100 * round(proba[0][1], 2)
            color = 'seagreen'
        else:
            text = 'Negative'
            class_proba = 100 * round(proba[0][0], 2)
            color = 'crimson'

        ax.text(0.5, 0.5, text, fontsize=50, ha='center', color=color)
        ax.text(0.5, 0.20, str(class_proba) + '%', fontsize=14, ha='center')
        ax.axis('off')
        ax.set_title('Sentiment Analysis', fontsize=14)

        # Display the plot using st.pyplot()
        st.pyplot(fig)