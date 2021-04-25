import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
from nltk.corpus import stopwords
from langdetect import detect
from encoding import split_data

## import dataset
data = pd.read_csv('data/sampledata.csv')
data = split_data(data)

# Encoding methods to transform the feature "reviews" str into for ex: One hot, etc as described in the initial report



# Learning methods to use the encoding feature and to classify the review, predict the number of stars based on the encoding feature
