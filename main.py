#===================Importing dependencies=================================#
import string
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from tqdm.notebook import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


from sklearn.preprocessing import MinMaxScaler

from flask import Flask
from flask_cors import CORS

import textstat

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.quantization
import torch.nn.utils.prune as prune
import requests

from scipy.special import softmax

from datasets import load_dataset
from evaluate import load as load_metric
from tensorflow.keras.callbacks import EarlyStopping


pd.options.display.memory_usage = 'deep'



dfmain = pd.read_csv('final_dataset.csv', encoding='latin1') #encoding because of special chars
dfmain['Id'] = range(len(dfmain))
cols = ['Id'] + [col for col in dfmain.columns if col != 'Id']
dfmain = dfmain[cols]

#plt.style.use('ggplot')
#======================Parameter A: Verification Check=====================#
verified = dfmain['verified']

verified_dict = []
for i in range(len(verified)):
    id = i
    verified_dict.append({'Id':id, 'verification_score':verified[i]})
    

verified = pd.DataFrame(verified_dict)
verified['invert_verification_score'] = 1 - verified['verification_score']  # 0 for verified, 1 for not



#======================Parameter B: Autoencoder for anomaly detection============================#
texts = dfmain['review'] 
tokenizer1 = Tokenizer(num_words = 10000)
tokenizer1.fit_on_texts(texts)
sequences = tokenizer1.texts_to_sequences(texts)
word_index = tokenizer1.word_index

max_len = 100  # Define a max length for padding
data_padded = pad_sequences(sequences, maxlen=max_len)

input_dim = data_padded.shape[1]  # Number of features in input

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

autoencoder.fit(data_padded, data_padded, epochs=50, batch_size=64, shuffle=True, validation_split=0.2, verbose = 1)


reconstructions = autoencoder.predict(data_padded)
mse = np.mean(np.power(data_padded - reconstructions, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)  # You can adjust the percentile based on your needs
anomalies = mse > threshold

# Flag potential fake reviews
fake_reviews = dfmain[anomalies]

anomaly_score_dict = []
for i in range(len(anomalies)):
    id = i
    score = 1 if anomalies[i] else 0 
    anomaly_score_dict.append({'Id':id, 'anomaly_score':score})
    

anomaly_score = pd.DataFrame(anomaly_score_dict)

#print(fake_reviews)

#======================Parameter C: Classifier=============================#


df = pd.read_csv('Classifier_dataset.csv', usecols = ["text_", "label_num"], dtype={"label_num" : "int8"})


df['text_'] = df['text_'].apply(lambda x: x.replace('\r\n', ' ')) 


#stemming 
stemmer = PorterStemmer()
corpus = []

stopwords_set = set(stopwords.words('english'))

for i in range(len(df)):
    text = df['text_'].iloc[i].lower()
    text = text.translate(str.maketrans('','',string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)


#vectorize 
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(corpus) 
y = df['label_num']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

clf = MultinomialNB()
clf.fit(x_train, y_train)
#print(f"Model accuracy: {accuracy_score(y_test, clf.predict(x_test))}")

arr_classifier = []
for i in range(len(dfmain)):
    to_classify = dfmain['review'].iloc[i].lower()
    to_classify = to_classify.translate(str.maketrans('','',string.punctuation)).split()
    to_classify = [stemmer.stem(word) for word in to_classify if word not in stopwords_set]
    to_classify = ' '.join(to_classify)
    review_corpus = [to_classify]
    x_review = vectorizer.transform(review_corpus)

    classifier_score = clf.predict(x_review)
    arr_classifier.append(classifier_score[0])
    
classifier_dict = []
for i in range(len(arr_classifier)):
    id = i
    classifier_dict.append({'Id':id, 'classifier_score':arr_classifier[i]})
    

classifier = pd.DataFrame(classifier_dict)

#======================Parameter D: Sentiment Analysis=============================#
#ax = dfmain['ratings'].value_counts().sort_index().plot(kind='bar', title='Review Count by Stars', figsize = (10,5))

#ax.set_xlabel('review star')
#ax.set_ylabel('Number')
#plt.show()
pd.set_option('display.max_columns', None)  # Show all columns 
pd.set_option('display.width', 1000)
sia = SentimentIntensityAnalyzer()
res = {}
Id=[]
for i in range(len(dfmain)):
    text = dfmain['review'].iloc[i]
    id = i
    Id.append(id)
    res[id] = sia.polarity_scores(text)

dfmain['Id'] = range(len(dfmain))
vaders_result = pd.DataFrame(res).T
vaders_result['Id'] = range(len(dfmain))
vaders_result = vaders_result.merge(dfmain, how='left')

vaders_result['normalized_sentiment_score'] =  vaders_result['compound'].apply(lambda x: 1 if abs(x) > 0.85 else 0)

#ax = sns.barplot(data = vaders_result, x = 'ratings', y='compound')
#ax.set_title('vader plot')
#plt.show()
#======================Parameter E: Helpful tag=========================================#
scaler = MinMaxScaler()
dfmain['helpful_score'] = scaler.fit_transform(dfmain[['helpful']])
dfmain['helpful_score'] = 1 - dfmain['helpful_score']



#=======================Parameter F: LLM INTEGRATION===================================#

tokenizer = BertTokenizer.from_pretrained('./fine-tuned-bert')
model = BertForSequenceClassification.from_pretrained('./fine-tuned-bert')

def get_authenticity_score(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).numpy()
    authenticity_score = scores[0][1]  # assuming the second class is "authentic"
    return authenticity_score

dfmain['authenticity_score'] = dfmain['review'].apply(get_authenticity_score)

dfmain['authenticity_score'].to_csv('authenticity_score.csv')
#======================================Main Function===========================================#

weights = {
    'verification_check' : 0.25,   
    'Autoencoder' : 0.2,         
    'Classifier' : 0.25,          
    'Sentiment_Score' : 0.10,
    'Helpful_Score' : 0.10,
    'llms_score' : 0.10
}


dfmain['FINAL_SCORE'] = (
    verified['invert_verification_score'] * weights['verification_check'] +
    anomaly_score['anomaly_score'] * weights['Autoencoder'] +
    classifier['classifier_score'] * weights['Classifier'] +
    vaders_result['normalized_sentiment_score'] * weights['Sentiment_Score'] +
    dfmain['helpful_score'] * weights['Helpful_Score'] +
    dfmain['authenticity_score'] * weights['llms_score']
)



dfmain['FINAL_SCORE'] = (dfmain['FINAL_SCORE'])*10000
dfmain['FINAL_SCORE'] = dfmain['FINAL_SCORE']//max(dfmain['FINAL_SCORE'])

print(dfmain['FINAL_SCORE'].max())






#============================================PROBLEM 2==================================================#
Product_url = [
    "https://www.amazon.co.uk/Earphones-In-ear-Black-Red-White/dp/B07166Q2LS/ref=cm_cr_srp_d_product_top?ie=UTF8",
    "https://www.amazon.co.uk/XuDirect-In-Ear-Earphones-Black-Green/dp/B07166Q1VW/ref=cm_cr_srp_d_product_top?ie=UTF8",
    "https://www.amazon.co.uk/AmazonBasics-E300-Sport-In-Ear-Headphones-Black/dp/B00L3KSRTW/ref=cm_cr_srp_d_product_top?ie=UTF8",
    "https://www.amazon.co.uk/XuDirect-Earphones-In-Line-Headphones-Red/dp/B0714MM6N3/ref=cm_cr_srp_d_product_top?ie=UTF8",
    "https://www.amazon.co.uk/Bluetooth-Speakers-XuDirect-Portable-Subwoofer-WHITE/dp/B01MQD4EWA/ref=cm_cr_srp_d_product_top?ie=UTF8",
    "https://www.amazon.co.uk/Amazon-Fire-TV-Stick-Streaming-Media-Player/dp/B00KAKUN3E/ref=cm_cr_srp_d_product_top?ie=UTF8"
]


proddf = pd.read_csv('product.csv')

weights_2 = {
    'rev_avg' : 0.3,   
    'Price_check' : 0.2,     
    'description_quality' : 0.3,          
    'Sentiment_Score' : 0.20,
}

#===========Check 1 : Review average=============#
results = []
colors = []
for url in Product_url:
    product_reviews = dfmain[dfmain['product_link'] == url]
    if not product_reviews.empty:
        average_score = product_reviews['FINAL_SCORE'].mean()


        results.append({'product_url': url, 'Final_score_avg':average_score})
    else:
        results.append({'product_url': url, 'Final_score_avg':None})

results_df = pd.DataFrame(results)
results_df['Final_score_avg'] = results_df['Final_score_avg']/100


#============Check 2 : Price Check================#
ppnp = np.array(proddf['Price'])
mean_price = np.mean(ppnp)
std_price = np.std(ppnp)

z_score = (ppnp-mean_price)/std_price
price_scores = -z_score

min_score = np.min(price_scores)
max_score = np.max(price_scores)
normalized_price = (price_scores-min_score)/(max_score-min_score)
results_df['normalized_price_z_score'] = normalized_price

#============Check 3 : Listing Quality============#

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in ('!', '.', ':', ',', ';', '?')])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

def lexical_diversity(tokens):
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0



def description_length(text):
    return len(text.split())

def calculate_usefulness(text):
    readability = textstat.flesch_reading_ease(text)
    return readability

proddf['tokens'] = proddf['Description'].apply(preprocess_text)
proddf['richness'] = proddf['tokens'].apply(lexical_diversity)
proddf['length'] = proddf['Description'].apply(description_length)
proddf['usefulness'] = proddf['Description'].apply(calculate_usefulness)
proddf['length_normalized'] = proddf['length']/proddf['length'].max()
proddf['usefulness_normalized'] = proddf['usefulness'] / proddf['usefulness'].max()

results_df['description_score'] = (1- proddf['richness']) + (1-proddf['length_normalized']) + (1-proddf['usefulness_normalized'])
results_df['description_score'] = results_df['description_score']/results_df['description_score'].max()
#============Check 4 : Review Sentiment ============#
sent = []
for url in Product_url:
    product_sentiment = dfmain[dfmain['product_link'] == url]
    product_sentiment = product_sentiment.copy()
    product_sentiment.loc[:, 'Compound'] = vaders_result['compound']
    sent.append(product_sentiment['Compound'].mean())


results_df['sentiment_score'] = np.array(sent)


#=================== Formula for calc===============================#
proddf['PRODUCT_SCORE'] =  (
    results_df['Final_score_avg'] * weights_2['rev_avg'] +
    results_df['normalized_price_z_score'] * weights_2['Price_check'] +
    results_df['description_score'] * weights_2['description_quality'] +
    results_df['sentiment_score'] * weights_2['Sentiment_Score'] 
)

proddf['PRODUCT_SCORE'] = proddf['PRODUCT_SCORE']*100
proddf['PRODUCT_SCORE'] = proddf['PRODUCT_SCORE'].astype(int)






#============================================Flask Int===================================================#
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def index():
    return str(dfmain['FINAL_SCORE'].iloc[-1])

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 5000, debug = True)