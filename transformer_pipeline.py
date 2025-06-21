# ... existing imports ...
from transformers import pipeline  # Importing the pipeline for sentiment analysis
import pandas as pd

#======================Parameter B: Sentiment Analysis=============================#
pd.set_option('display.max_columns', None)  # Show all columns 
pd.set_option('display.width', 1000)

# Initialize the sentiment analysis pipeline
sent_pipeline = pipeline("sentiment-analysis")

res = {}
Id = []
for i in range(len(dfmain)):
    text = dfmain['review'].iloc[i]
    id = i
    Id.append(id)
    # Use the transformer pipeline for sentiment analysis
    sentiment_result = sent_pipeline(text)[0]  # Get the first result
    res[id] = {
        'label': sentiment_result['label'], 
        'score': sentiment_result['score']
    }

dfmain['Id'] = range(len(dfmain))
vaders_result = pd.DataFrame(res).T
vaders_result['Id'] = range(len(dfmain))

# Normalize sentiment score based on the label
vaders_result['normalized_sentiment_score'] = vaders_result['label'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

#