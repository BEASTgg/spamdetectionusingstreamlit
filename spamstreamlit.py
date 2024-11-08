import streamlit as st
import imaplib
import email
from email.header import decode_header
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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

def fetch_emails_from_gmail(username, password, folder='inbox', n_emails=5):
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.select(folder)
        
        status, messages = mail.search(None, 'ALL')
        email_ids = messages[0].split()
        
        fetched_emails = []
        for i in email_ids[:n_emails]:
            status, msg_data = mail.fetch(i, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()
                    fetched_emails.append(subject + " " + body)
        mail.close()
        mail.logout()
        return fetched_emails
    except Exception as e:
        st.error(f"Error: {e}")
        return []

def classify_emails(emails):
    results = []
    for email_text in emails:
        email_text_clean = re.sub(r'\W+', ' ', email_text)
        transformed_text = tfidf_vectorizer.transform([email_text_clean])
        prediction = classifier.predict(transformed_text)
        label = "Spam" if prediction[0] == 1 else "Ham"
        results.append(label)
    return results

st.title("Email Spam Classifier")
st.write("Enter your Gmail credentials below:")

username = st.text_input("Email", type="default")
password = st.text_input("Password", type="password")
if st.button("Fetch and Classify Emails"):
    if username and password:
        emails = fetch_emails_from_gmail(username, password)
        if emails:
            classifications = classify_emails(emails)
            for idx, (email_text, classification) in enumerate(zip(emails, classifications)):
                st.write(f"**Email {idx + 1}:** {classification}")
                st.write(f"Preview: {email_text[:200]}...") 
        else:
            st.write("No emails found or unable to fetch emails.")
    else:
        st.warning("Please enter both email and password.")
