**Restaurant Reviews Sentiment Analysis**

A machine learning project for classifying customer reviews as positive or negative using Natural Language Processing (NLP).

**Project Overview**

This project develops a sentiment classification model for restaurant reviews. It helps businesses understand customer feedback and take necessary actions to improve their services.

The model is trained on historic restaurant reviews and used to predict sentiment for a fresh dataset of reviews.




**Machine Learning Techniques Used**

Text Preprocessing (removing numbers, special characters, stopwords)
Feature Extraction using Bag of Words (BoW)
Naïve Bayes Classifier for sentiment classification
Performance Evaluation using accuracy score
Model Deployment using .pkl files for prediction


**Project Files & Explanation**

1. Datasets
📂 a1_RestaurantReviews_HistoricDump.tsv

Contains historic restaurant reviews.
Used for training the sentiment classification model.

📂 a2_RestaurantReviews_FreshDump.tsv

Contains 100 new restaurant reviews.
The trained model predicts sentiment for these reviews.

📂 c3_Predicted_Sentiments_Fresh_Dump.tsv

Stores predicted sentiment labels (positive/negative) for fresh reviews.
Helps businesses analyze customer feedback trends.

2.** Jupyter Notebooks**


📒 b1_Sentiment_Analysis_Model.ipynb

Main notebook for model training.
Steps included:
Data cleaning and preprocessing
Converting text into numerical features using BoW
Training the Naïve Bayes classifier
Evaluating model performance

📒 b2_Sentiment_Predictor.ipynb

Notebook for predicting sentiment on new reviews.
Loads trained model and predicts sentiments for a2_RestaurantReviews_FreshDump.tsv.
Saves predictions in c3_Predicted_Sentiments_Fresh_Dump.tsv.
3. Machine Learning Models

📁 c1_BoW_Sentiment_Model.pkl

Vectorizer model (Bag of Words).
Converts text reviews into numerical vectors for classification.

📁 c2_Classifier_Sentiment_Model

Trained Naïve Bayes model used for prediction.
Classifies reviews as positive (1) or negative (0).

4. Business Presentation
📄 d_Business_Deck_Sentiment_Analysis.pdf

Final report for business stakeholders.
Contains:
Model performance summary
Insights from sentiment predictions
Key business action points



The predicted results will be saved in c3_Predicted_Sentiments_Fresh_Dump.tsv.


**Results**

Model Accuracy: 72.8%

