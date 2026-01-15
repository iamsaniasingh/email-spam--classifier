import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# ---- TEXT CLEANER ----
def clean_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)
result_box = st.empty()

# ---- LOAD MODEL + VECTORIZER ----
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="🎯Spam Detector",
    page_icon="📬",
    layout="centered",
    initial_sidebar_state="auto",
)

# ---- SIDEBAR ----
with st.sidebar:
    st.title("📌 About")
    st.write("This is a EMAIL/SMS Spam Classifier built with NLP + Machine Learning that 🔥 automatically tells you whether a message is Spam or Not Spam. It’s all about saving your time and sanity — no one wants that sus clutter in their inbox! 😤📩 This dope classifier uses natural language processing to understand text patterns and a trained ML model to make decisions. Just type any message, hit Predict, and let the model call it out 🚨 — spam or not spam (legit). It’s like having your own digital bouncer for messages .")
    st.markdown("---")
    st.write("Made by Sania Singh")
    st.write("Enter any message & hit *Predict*")

# ---- MAIN UI ----
st.markdown("<h1 style='text-align: center;'>📩 Spam / Not Spam Detector</h1>", unsafe_allow_html=True)
st.write("Type anything & let the model decide if it's sus 👀")

input_sms = st.text_area("💬 **Enter Message**", height=120)

st.markdown("---")

if st.button("Predict 🚀"):
    
    if not input_sms.strip():
        st.error("🛑 Message cannot be empty 😑")
    else:
        with st.spinner("🧠 Thinking..."):
            new_clean = clean_text(input_sms)
            vector_input = tfidf.transform([new_clean]).toarray()
            result = model.predict(vector_input)[0]

            if result == 1:
                st.markdown(
                    "<h2 style='text-align: center; color: red;'>🔥 SPAM</h2>", 
                    unsafe_allow_html=True
                )
                st.write("⚠️ Be careful — this looks like spam.")
            else:
                st.markdown(
                    "<h2 style='text-align: center; color: green;'>✔️ NOT SPAM</h2>", 
                    unsafe_allow_html=True
                )
                st.write("🎉 Legit message! Safe to read.")
