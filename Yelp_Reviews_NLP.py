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


"""
Visualizing the dataset to better understand the dataset. We will count the words 
in the dataset and add it in a new column in our dataset.
"""
# Creating a column for length of reviews
yelp_reviews['length'] = yelp_reviews['text'].apply(len) # Using the .apply(len) function to find the length of every review in the text column
yelp_reviews # Shows the newly added column

# Creating a histogram for the length column
sns.histplot(yelp_reviews['length'], bins=150, color='orange', kde=True)
plt.title("Length of All Reviews")
plt.xlabel("Length of Reviews")
plt.ylabel("Number of Reviews")


# Exploring the max and min length reviews
yelp_reviews[yelp_reviews['length'] == 4997]['text'].iloc[0]
yelp_reviews[yelp_reviews['length'] == 1]['text'].iloc[0]


# Plotting the star count for reviews
sns.count(y = 'stars', data = yelp_reviews)

facet = sns.FacetGrid(data = yelp_reviews, col = 'stars', col_wrap = 3)
facet.map(sns.histplot, 'length', bins = 20, color = 'orange') # Created facet graphs for each group of star counts (1 to 5) and their lengths, we can see that the length increases as the number of stars increases

# Create 1 and 5 star datsets
yelp_reviews_1star = yelp_reviews[yelp_reviews['stars'] == 1]
yelp_reviews_1star

yelp_reviews_5star = yelp_reviews[yelp_reviews['stars'] == 5]
yelp_reviews_5star


# Concatenating the 1 and 5 star datasets
yelp_reviews_1_5stars = np.concat([yelp_reviews_1star, yelp_reviews_5star])
yelp_reviews_1_5stars


# Printing their percentage values
print("1-Star Reviews = ", round((len(yelp_reviews_1star) / len(yelp_reviews_1_5stars)) * 100, 2), "%")

print("5-Star Reviews = ", round((len(yelp_reviews_5star) / len(yelp_reviews_1_5stars)) * 100, 2), "%")


# Plotting the 1 and 5 Star reviews
sns.countplot(x = yelp_reviews_1_5stars['stars'], data = yelp_reviews_1_5stars, label = 'Count')


"""
To prepare the data for training we need to perform so fundamental natural language
processing techniques for the dataset to be ready. These techniques clean the data
and allow the model to interpret it without any unnecessary information clouding 
the model's performance. 

The model is able to interpret data from a matrix of words and their frequency 
count called a count vectorizer. The count vectorizer essentially contains a list
of all uncommon words and numbers, as well as how often they occur in each review. 
For example, if the word "excellent" is used 3 times in a review, the model can 
most likely predict that the review is a highly rated review. 

For the count vectorizer to work properly, the text needs to be cleaned to remove 
any punctuation or common (meaningless) words aka stopwords. To remove punctuation
we will run a for loop to check every character in every review and only keep the
characters that are not in our list of punctuation. 

Next, the meaningless stop words need to be removed by repeating the same process 
as before. Use a for loop to check if the word is in our list of stop words, and 
then only keeping the ones not in the list. 

Finally, we use the cleaned text to create a count vectorizer from our sci-kit 
learn library class. The count vectorizer will fit around our clean data and then
transform into a matrix of words and frequencies. 
"""
# Removing punctuation
import string
string.punctuation

def removing_punc(message):
  Text_punc_removed = []

  for char in message:
    if char not in string.punctuation:
      Text_punc_removed.append(char)

  Text_punc_joined = "".join(Text_punc_removed)
  return Text_punc_joined


yelp_nopunc = yelp_reviews['text'].apply(removing_punc)
yelp_nopunc


# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

len(stopwords.words('english'))

def removing_stopwords(message):
  Text_clean = []

  for word in message.split():
    if word.lower() not in stopwords.words('english'):
      Text_clean.append(word)
  return Text_clean

yelp_clean = yelp_nopunc.apply(removing_stopwords)
print(yelp_clean[0]) # Verifying the first cleaned up review has no punctuation or stopwords


"""
Finally, we use the cleaned text to create a count vectorizer from our sci-kit 
learn library class. The count vectorizer will fit around our clean data and then 
transform into a matrix of words and frequencies. It converts text into numerical
representation by creating a matrix that shows the frequency in occurences of 
tokenized words. The text input is tokenized into words or n-grams (small groups 
of words) and then counts each occurence of that group.
"""
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer() # Importing the CountVectorizer class from sklearn and creating a vectorizer instance object under the class

yelp_vectorizer = vectorizer.fit_transform([" ".join(i) for i in yelp_clean]) # Use the .join() method to join every word in the yelp_clean list back into a review (the words were seperated so we could iterate through them). Then use the .fit_transform() method from the vectorizer class to fit the data to the vectorizer and create the count vectorizer matrix, store the results in yelp_vectorizer

print(vectorizer.get_feature_names_out())
print(len(vectorizer.get_feature_names_out()))
print(yelp_vectorizer.shape) # Our count vectorizer matix has 21,882 unique words from 4,086 reviews.


"""
Applying multinomial naive bayes classification algorithm which is commonly used
in text classification because it models the feature as multinomials, or discrete
distributions, with the word frequencies (stored in the count vectorizer). The 
naive bayes algorithms are popular because they are computationally efficient,
which is great for text classification, spam filtering, and sentiment analysis.
"""
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB() # Importing the naive bayes classifier we will be using for this project from sci-kit learn and then creating the object NB_classifier from the class

label = yelp_reviews_1_5stars['stars'].values
label # Assigning the variable label to the values in the stars column. This means that label is an array containing star count

# Training on entire dataset (for fun)
NB_classifier.fit(yelp_vectorizer, label)
sample = [input()] # Creating a fun section to test my own reviews on here, it is good at detecting obvious ones but sometimes it gives wrong answers for some of the reviews that are clearly wrong. For example, try typing "I hated eating here, the food was bad" which sounds like a one star review but it thinks it's a four star review

testing_sample = vectorizer.transform(sample)
sample_predict = NB_classifier.predict(testing_sample)

print(sample_predict) 


"""
Splitting the dataset into training and testing sets so we can evaluate the model's
performance after training. 
"""
X = yelp_vectorizer
X.shape # Our features dataset (X) will be the count vectorizer which stores the frequency of all the uncommon words 4,806 reviews

y = label
y.shape # The dependent variable (y) is the labeled star count for every review

from sklearn.model_selection import train_test_split # Importing the train_test_split function to split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # Splitting the data into training and testing sets for features and dependent variables


"""
Train the model on the training dataset that we created previously and fit the 
multinomial naive bayes classifier to the model
"""
from sklearn.naive_bayes import MultinomialNB
NB_classfier = MultinomialNB()

NB_classifier.fit(X_train, y_train) # Fitting the naive bayes classifier to the training dataset


"""

"""