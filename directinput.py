import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

st.title("Email Spam Classifier")

email_input = st.text_area("Paste the email content here to classify:")

if st.button("Classify"):
    if email_input:
        email_text_clean = re.sub(r'\W+', ' ', email_input)
        transformed_text = tfidf_vectorizer.transform([email_text_clean])
        prediction = classifier.predict(transformed_text)
        label = "Spam" if prediction[0] == 1 else "Ham"
        st.write(f"The email is classified as: **{label}**")
    else:
        st.warning("Please paste an email to classify.")
