from nltk.corpus import stopwords
from gensim.models import Word2Vec
import re
import string
import pandas as pd
from collections import defaultdict
import spacy
from sklearn.manifold import TSNE
# import nltk
# nltk.download('stopwords')
# STOPWORDS = set(stopwords.words('english'))

df = pd.read_csv('./data/emoji_dataset_new.csv')


def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove a sentence if it is only one word long
#     if len(text) > 2:
#         return ' '.join(word for word in text.split() if word not in STOPWORDS)


df_clean = pd.DataFrame(df.text.apply(lambda x: clean_text(x)))
print(df_clean)
