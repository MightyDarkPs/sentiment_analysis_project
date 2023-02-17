import pickle
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title('Sentiment Analysis')
st.markdown("""
Made by Toktaganov Turlykhan BDA-2106
""")

def predict(vectoriser, model, text):
    processedText = []

    wordLemm = WordNetLemmatizer()

    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in text:
        tweet = tweet.lower()

        tweet = re.sub(urlPattern,' URL',tweet)
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        tweet = re.sub(userPattern,' USER', tweet)
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            if len(word)>1:
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')

        processedText.append(tweetwords)

    textdata = vectoriser.transform(processedText)
    sentiment = model.predict(textdata)

    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))

    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

nltk.download('wordnet')
nltk.download('omw-1.4')

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

file = open('vectoriser.pickle', 'rb')
vectoriser = pickle.load(file)
file.close()

file = open('model.pickle', 'rb')
model = pickle.load(file)
file.close()

text = st.text_input('Text here: ')

df = predict(vectoriser, model, text)

st.write('Here is our sentiment analysis')
print(df.head())