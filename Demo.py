from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
def transform_text(text):
    # 01: transforming text into lower case
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # 02: getting alphnumeric content from text
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)
    
    # 03: removing stop words and punction marks from text
    text = y[:]
    y.clear()
    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(word)
            
    # 04: apply stemming 
    text = y[:]
    y.clear()
    for word in text:
        y.append(PorterStemmer().stem(word))
    return " ".join(y)


def demoDecisionTree(model, inp):
    df = pd.read_csv("DataEmail.csv")
    demo = transform_text(inp)
    demo_tf_idf = TfidfVectorizer(max_features=1000)
    demo_df = df.copy(deep=False)
    length_df = len(demo_df)
    demo_df.loc[length_df] = demo
    demo_x = demo_tf_idf.fit_transform(demo_df["transformed_text"]).toarray()
    demo_y = model.predict([demo_x[length_df]])
    return demo_y

loaded_model = joblib.load('decision_tree_model.pkl')
while(True):
    print("Nhap gmail: ")
    inp = input()
    print(demoDecisionTree(loaded_model, inp))
    