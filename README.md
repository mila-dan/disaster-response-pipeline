# Disaster Response Pipeline

## by Liudmila Danilovskaya 

The goal of this project is to create a machine learning pipeline to categorize messages that were sent during disaster events.  The project includes a web app where an emergency worker can input a new message and get classification results in several categories.

## Datasets 

1. disaster_categories.csv - a dataset containing the messages ids and their categories
	
1. disaster_messages.csv - a dataset containing real messages that were sent during disaster events



## Installation and running

This project requires Python 3.x and the following Python libraries installed:

pandas
scikit-learn
plotly
nltk
sqlalchemy
flask

All libraries can be installed using the command: pip install -r requirements.txt 

To run the project:

1. python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

	This creates a sqlite database DisasterResponse.db using input files "disaster_messages.csv" and "disaster_categories.csv" and a final ETL script process_data.py. 

2. python train_classifier.py ../data/DisasterResponse.db classifier.pkl
	
	This builds a machine learning model classifier.pkl using a python machine learning script train_classifier.py.

3. python run.py

	This runs a Flask web app where using can enter a new message to classify and see the input dataset overview. 


## Summary of Findings

3 machine learning models have been tested:
1. RandomForestClassifier gives a high training accuracy (0.99) but low test accuracy (0.27%)
2. MultinomialNB gives low training and test score
3. Logistic Regression gives the highest testing score (0.32%) out of all models

Logistic Regression model has been chosen for this dataset. The parameters tuning was done using GridSearchCV which gave the following best parameters:
{'clf__estimator__C': 1.0,
 'clf__estimator__penalty': 'l1',
 'clf__estimator__solver': 'saga',
 'text_pipeline__vect__max_df': 0.85}


## Acknowledgments

This project was done as the part of Udacity learning. 