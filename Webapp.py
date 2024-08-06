import streamlit as st
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import warnings
import re
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
import nltk

# import sklearn
# print(sklearn.__version__)
@st.cache_data
def download_stopwords():
    try:
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords')

@st.cache_data
def load_model():
    with open('LogisticRegModel2.pkl', 'rb') as file:
        return pickle.load(file)


LR_pipeline = load_model()
stemmer = SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

@st.cache_data
def remove_stopwords(text):
    text_without_stopwords = [i for i in text.split() if i not in stopwords]
    return " ".join(text_without_stopwords)

@st.cache_data
def stemming(sentence):
    stemmed_sentence = ""
    for word in sentence.split():
        stemmed_word = stemmer.stem(word)
        stemmed_sentence += stemmed_word + " "

    stemmed_sentence.strip()
    return stemmed_sentence

@st.cache_data
def text_cleaning(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", "not", text)
    text = re.sub(r"i'm", "i am ",text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'11", " will", text)
    text = re.sub(r"\'scuse", " excuse", text)
    text = re.sub('\W', ' ',text)
    text = re.sub('\s+',' ', text)
    text = text.strip('')

    return text



st.title('Toxic Comments - Multi Label Classification')
st.text("The Model Clasasifies the provided text and tags it the following:")
st.text("toxic, severe_toxic, obscene, threa, insult, identity_hate")
user_input = st.text_area("Enter text for classification:")


tags = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

if st.button("Classify"):
    if user_input:
        with st.spinner('Processing...'):

            cleaned = text_cleaning(remove_stopwords(user_input))
            stemmed = stemming(cleaned)

            prediction = LR_pipeline.predict([stemmed])[0]

            result_tags = [tag for tag, pred in zip(tags, prediction) if pred == 1]
            if result_tags:
                st.write(f"Tags: {', '.join(result_tags)}")
            else:
                st.write("NON TOXIC")
    else:
        st.write("Please enter text to classify.")



# Metrics

roc_auc = 0.9789765338334316
accuracy = 0.9181889393702021
classification_rep = """
               precision    recall  f1-score   support

        toxic       0.91      0.62      0.74      3056
 severe_toxic       0.58      0.27      0.37       321
      obscene       0.91      0.64      0.75      1715
       threat       0.62      0.14      0.22        74
       insult       0.81      0.51      0.63      1614
identity_hate       0.70      0.16      0.26       294

    micro avg       0.87      0.56      0.68      7074
    macro avg       0.75      0.39      0.50      7074
 weighted avg       0.86      0.56      0.67      7074
  samples avg       0.06      0.05      0.05      7074
"""

# Streamlit app
st.title('Model Performance Metrics')

# Create a grid layout for metrics
col1, col2 = st.columns(2)

with col1:
    st.metric(label="ROC-AUC", value=f"{roc_auc:.4f}")
    st.metric(label="Accuracy", value=f"{accuracy:.4f}")

with col2:
    st.write("Classification Report:")
    st.text(classification_rep)









st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        color: #555;
    }
    </style>
    <div class="footer">
        <p>Created by Kushaagra</p>
    </div>
    """, unsafe_allow_html=True)


