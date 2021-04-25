import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.gridspec as gridspec
import numpy as np

business_json_path = 'data/yelp_academic_dataset_business.json'  # path to dataset
data_business = pd.read_json(business_json_path, lines=True, nrows=1000)

business_Restaurants = data_business[data_business['categories'].str.contains(
              'Restaurants|Pop-Up Restaurants', case=False, na=False)]

review_json_path = 'data/yelp_academic_dataset_review.json'
size = 100000
data_reviews = pd.read_json(review_json_path, lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                      chunksize=size)

# There are multiple chunks to be read
chunk_list = []
for chunk_review in data_reviews:
    # Drop columns that aren't needed
    chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1)
    # Renaming column name to avoid conflict with business overall star rating
    chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
    # Inner merge with edited business file so only reviews related to the business remain
    chunk_merged = pd.merge(business_Restaurants, chunk_review, on='business_id', how='inner')
    # Show feedback on progress
    print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
    chunk_list.append(chunk_merged)

# After trimming down the review file, concatenate all relevant data back to one dataframe
dataset = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)

# Encoding methods to transform the feature "reviews" str into for ex: One hot, etc as described in the initial report



# Learning methods to use the encoding feature and to classify the review, predict the number of stars based on the encoding feature
