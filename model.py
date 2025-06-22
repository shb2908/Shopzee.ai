#===================Importing dependencies=================================#
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

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import CLIPProcessor, CLIPModel

import os
from PIL import Image

pd.options.display.memory_usage = 'deep'

"""
AMAZON PRODUCT AUTHENTICITY DETECTION SYSTEM - VLM & LLAMA-2 VERSION
===================================================================

This system integrates VLM and Llama-2 models to detect fake products and reviews:

1. VERIFICATION CHECK: Basic verification status
2. SENTIMENT ANALYSIS: VADER sentiment scoring
3. HELPFUL SCORE: User helpfulness metrics
4. VLM (CLIP): Vision-Language Model for image analysis
5. Time series analysis to detect spam reviews
6. LLAMA-2: Large Language Model with a classifier head for sentiment analysis of reviews for authenticity check.

The final score is a weighted combination of all these parameters.
"""

#=======================DATA LOADING===========================================#

dfmain = pd.read_csv('final_dataset.csv', encoding='latin1') #encoding because of special chars
dfmain['Id'] = range(len(dfmain))
cols = ['Id'] + [col for col in dfmain.columns if col != 'Id']
dfmain = dfmain[cols]

#======================Parameter A: Verification Check=====================#
verified = dfmain['verified']
verified_dict = []
for i in range(len(verified)):
    id = i
    verified_dict.append({'Id':id, 'verification_score':verified[i]})
verified = pd.DataFrame(verified_dict)
verified['invert_verification_score'] = 1 - verified['verification_score']  # 0 for verified, 1 for not

#======================Parameter B: Sentiment Analysis=============================#
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
vaders_result['normalized_sentiment_score'] = vaders_result['compound'].apply(lambda x: 1 if abs(x) > 0.85 else 0)

#======================Parameter C: Helpful tag=========================================#
scaler = MinMaxScaler()
dfmain['helpful_score'] = scaler.fit_transform(dfmain[['helpful']])
dfmain['helpful_score'] = 1 - dfmain['helpful_score']

#=======================Parameter D: VLM INTEGRATION===================================#
# Vision-Language Model for product image analysis using CLIP

class CLIPClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.clip = base_model
        self.classifier = nn.Linear(base_model.config.projection_dim, 2)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=False,
        )
        img_embeds = outputs.image_embeds
        return self.classifier(img_embeds)

def get_vlm_score(image_path, text_prompt="real product"):
    """
    Get VLM-based authenticity score for product images using CLIP
    Returns a score between 0 and 1, where 1 indicates authentic product
    """
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        model = CLIPClassifier(clip_model)
        with torch.no_grad():
            outputs = model(
                inputs["input_ids"],
                inputs["attention_mask"], 
                inputs["pixel_values"]
            )
            scores = torch.softmax(outputs, dim=1)
            vlm_score = scores[0][1].item()  # Probability of being authentic
        return vlm_score
    except Exception as e:
        print(f"VLM processing error: {e}")
        return 0.5  # Default neutral score

# Apply VLM scoring to products with images if image_path column exists
if 'image_path' in dfmain.columns:
    dfmain['vlm_score'] = dfmain['image_path'].apply(lambda x: get_vlm_score(x) if pd.notnull(x) else 0.5)
else:
    dfmain['vlm_score'] = 0.5  # Default neutral score for products without images

# Save VLM scores
dfmain['vlm_score'].to_csv('vlm_scores.csv')

#=======================Parameter E: Time Series Spam Review Detection==================#
def detect_spam_reviews(df, time_col='reviewTime', user_col='reviewerID', threshold_seconds=60):
    """
    Flags reviews as spam if the same user posts multiple reviews for the same product within a short time window.
    Adds a 'spam_flag' column: 1 if spam, 0 otherwise.
    """
    if time_col not in df.columns or user_col not in df.columns:
        df['spam_flag'] = 0
        return df

    df = df.sort_values([user_col, time_col])
    df['reviewTime_dt'] = pd.to_datetime(df[time_col], errors='coerce')
    df['prev_review_time'] = df.groupby(user_col)['reviewTime_dt'].shift(1)
    df['time_diff'] = (df['reviewTime_dt'] - df['prev_review_time']).dt.total_seconds()
    df['spam_flag'] = ((df['time_diff'] < threshold_seconds) & (df['time_diff'].notnull())).astype(int)
    df = df.drop(['reviewTime_dt', 'prev_review_time', 'time_diff'], axis=1)
    return df

dfmain = detect_spam_reviews(dfmain)

spam_weight = 0.10
if 'spam_flag' in dfmain.columns:
    dfmain['spam_penalty'] = dfmain['spam_flag'] * spam_weight
else:
    dfmain['spam_penalty'] = 0

#=======================Parameter F: LLAMA-2 INTEGRATION===================================#
def setup_llama_model():
    """
    Setup Llama-2 model for fine-tuning and inference
    """
    MODEL_CHECKPOINT = "meta-llama/Llama-2-7b-hf"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model
    except Exception as e:
        print(f"Llama-2 model loading error: {e}")
        return None, None

def get_llama_score(text, tokenizer, model):
    """
    Get Llama-2 based authenticity score for text
    Returns a score between 0 and 1, where 1 indicates authentic content
    """
    try:
        inputs = tokenizer(
            text, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True, 
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            llama_score = scores[0][1].item()  # Probability of being authentic
        return llama_score
    except Exception as e:
        print(f"Llama-2 processing error: {e}")
        return 0.5  # Default neutral score

llama_tokenizer, llama_model = setup_llama_model()
if llama_model is not None:
    dfmain['llama_score'] = dfmain['review'].apply(
        lambda x: get_llama_score(x, llama_tokenizer, llama_model)
    )
else:
    dfmain['llama_score'] = 0.5  # Default neutral score if model not available

dfmain['llama_score'].to_csv('llama_scores.csv')

#======================================Main Function===========================================#

weights = {
    'verification_check' : 0.22,   
    'Sentiment_Score' : 0.13,
    'Helpful_Score' : 0.13,
    'vlm_score' : 0.18,
    'llama_score' : 0.24
}

dfmain['FINAL_SCORE'] = (
    verified['invert_verification_score'] * weights['verification_check'] +
    vaders_result['normalized_sentiment_score'] * weights['Sentiment_Score'] +
    dfmain['helpful_score'] * weights['Helpful_Score'] +
    dfmain['vlm_score'] * weights['vlm_score'] +
    dfmain['llama_score'] * weights['llama_score'] -
    dfmain['spam_penalty']
)

# Save final results
dfmain[['Id', 'review', 'FINAL_SCORE']].to_csv('final_results_vlm_llama.csv', index=False)

print("VLM and Llama-2 integration completed!")
print(f"Total products analyzed: {len(dfmain)}")
print(f"Average authenticity score: {dfmain['FINAL_SCORE'].mean():.4f}")
print(f"Products flagged as potentially fake: {(dfmain['FINAL_SCORE'] > 0.7).sum()}")

#=======================Flask API Setup===========================================#

from flask import Flask, jsonify
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Amazon Product Authenticity Detection API - VLM & Llama-2 Version"

@app.route("/analyze/<int:product_id>")
def analyze_product(product_id):
    if product_id < len(dfmain):
        product = dfmain.iloc[product_id]
        return jsonify({
            'product_id': product_id,
            'review': product['review'],
            'final_score': float(product['FINAL_SCORE']),
            'vlm_score': float(product['vlm_score']),
            'llama_score': float(product['llama_score']),
            'verification_score': float(verified.iloc[product_id]['invert_verification_score']),
            'sentiment_score': float(vaders_result.iloc[product_id]['normalized_sentiment_score']),
            'helpful_score': float(product['helpful_score']),
            'spam_flag': int(product['spam_flag']) if 'spam_flag' in product else 0
        })
    else:
        return jsonify({'error': 'Product ID not found'}), 404

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)