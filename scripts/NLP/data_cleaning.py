import pandas as pd
import numpy as np
import json
from IPython.display import display, HTML

pd.set_option('display.max_colwidth', 100)

data = []
# number of records (per label) to keep for training/testing
data_count = 10000

# counters
count_p=0
count_n =0

# helper variable
to_enter =0


# only keep 1, 2, 4, 5 and create balanced dataset by equalizing positive and negative records
with open('./review.json', encoding="utf8") as f:
    for line in f:
        review = json.loads(line)
        raw_rating = review['stars']        
        
        if(raw_rating == 3):
            continue
        
        if((raw_rating == 1 or raw_rating == 2) and count_n < data_count):
            sentiment = "negative"
            count_n+=1
            to_enter =1           
        elif((raw_rating == 4 or raw_rating == 5) and count_p < data_count):
            sentiment = "positive"
            count_p+=1
            to_enter = 1
            
        if(to_enter == 1):
            to_enter = 0
            entry = {
                "review": review['text'],
                "sentiment": sentiment
            }
            data.append(entry)
        
        if(count_p >= data_count and count_n >= data_count):
            break
            
df = pd.DataFrame(data)
#display(df.head(10))
#display(df.shape)