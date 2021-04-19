import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np


business_json_path = 'data/yelp_academic_dataset_business.json' #path to dataset
data_business = pd.read_json(business_json_path, lines=True)

