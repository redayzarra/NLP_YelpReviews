"""
Yelp Reviews Classification

This project covers the topic of natural language processing or NLP to classify 
user-generated text and determine their intent. The goal of this project is to build
a model that can automatically classify 10,000 Yelp reviews into one of several 
predefined categories, such as "positive," "negative," or "neutral." To accomplish 
this, the project uses NLP techniques to process and analyze the text of the 
reviews.

The model can then be used to predict the label for new, unseen reviews. This can 
be especially useful for businesses, as it can help them identify patterns in 
customer feedback and make improvements to their products or services.

We will start by importing the necessary libraries for natural language processing:
"""
# Importing the necessary libraries
import pandas as pd # Pandas is used for data frame manipulations
import numpy as np # NumPy is a package used for numerical analysis

import matplotlib.pyplot as plt
import seaborn as sns

"""
Importing the dataset with the .read_csv method from Pandas to load the dataset. 
"""
yelp_reviews = pd.read_csv('yelp.csv') # Importing the dataset with the .read_csv method from Pandas to load the dataset and storing it in the yelp_reviews variable.

yelp_reviews.head() # The .head() method is used to show the first five rows of the dataframe.
yelp_reviews.tail() # The .tail() method is used to show the last five rows of the dataframe.

yelp_reviews.describe() # The .describe() method from Pandas gives us a summary of what the dataset contains. The count tells us the number of values in the dataset for each feature so 10,000 total reviews. The mean tells us the mean of each column, for example the mean for the stars is 3.78 while 0.88 is the mean for the cool votes column. The std, or standard deviation, column is 1.2 which is the dispersion around the mean. The smallest value for each dataset is stored in the min column, while the maximum is stored in the max. Finally, the quartiles are represented by the 25%, 50% (median), and 75%
yelp_reviews.info() # There are no missing values in our dataset, which is a great thing because our model has more data to work with

print(yelp_reviews['text'][0]) # Allows us to target the 'text' column and access the first review inside