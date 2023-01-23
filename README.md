# Yelp Reviews Classification - Natural Language Processing

## Overview
This project covers the topic of natural language processing or NLP to classify user-generated text and determine their intent. The goal of this project is to build a model that can classify 10,000 Yelp reviews into either one-star or 5-star reviews. To accomplish this, the project uses NLP techniques to process and clean the text from the reviews, and then train the model using the clean text. The model is then evaluated and analyzed to measure accuracy and performance.

The model is trained on this [dataset](https://github.com/redayzarra/ml-yelpreviews-project/blob/master/yelp.csv), and divided into training and testing sets. The model was originally trained on all 10,000 reviews to classify the reviews to their corresponding star count, however, this proved difficult as the reviews were often meaningless and unpredictable. This project utilizes the data to classify the reviews in either one-star or five-star reviews, the two extremes of the dataset.

The model can then be used to predict the label for new, unseen reviews. This can be especially useful for businesses, as it can help them identify patterns in customer feedback and make improvements to their products or services.

<div align="center">

<img src="https://user-images.githubusercontent.com/113388793/212522711-05d0fa8e-4abc-4ec9-bb8f-d96c8789fc27.png" width="700" height="600">

</div>


## Project Website

If you would like to find out more about the project, please checkout: [Yelp Reviews Classification Project](https://www.redaysblog.com/machine-learning/yelp-reviews)

## Installing the libraries

This project uses several important libraries such as Pandas, NumPy, Matplotlib, and more. You can install them all by running the following commands with pip:

```bash 
pip install pandas
pip install numpy

python -m pip install -U matplotlib
pip install seaborn

pip install -U scikit-learn
pip install tensorflow

```

If you are not able to install the necessary libraries, I recommend you **use Jupyter Notebook with Anaconda**. I have a .ipynb file for the project as well.


## Configurations

This project utilizes a CSV file for loading the dataset. If you have a CSV file full of text that you would like to use, please feel free to use this code to load your dataset in to the file:

```python
dataset = pd.read_csv('YOUR-DATA.csv')
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
