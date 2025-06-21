from transformers import pipeline
import numpy as np 
import pandas as pd

sent_pipeline = pipeline("sentiment-analysis")


dfmain = pd.read_csv('final_dataset.csv')



res = {}
for i in range(len(dfmain)):
    text = dfmain['review'].iloc[i]
    id = i
    res[id] = sent_pipeline(text)
    
dfmain['Id'] = range(len(dfmain))
result = pd.DataFrame(res)
print(result)