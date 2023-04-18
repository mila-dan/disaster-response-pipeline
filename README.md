# Disaster Response Pipeline

## by Liudmila Danilovskaya 

The goal of this project is to create a solution which allows to effectively response to a disaster by classifying the text messages received during the disaster and further sending them to the relevant disaster response organizations. 

This is achieved by creating a machine learning model that categorizes thousands of real messages that were sent during natural disaster events into one or multiple categories. The user can input a new message in the web app and get the categories of the message. 

## Datasets 

1. disaster_categories.csv - a dataset containing the messages ids and their categories
	
1. disaster_messages.csv - a dataset containing real messages that were sent during disaster events

## Project structure

+ app
	+ template					
		+ master.html: main page of web app
		+ go.html: classification result page of web app
	+ run.py: Flask file that runs app
+ data
	+ disaster_categories.csv: categories data to process
	+ disaster_messages.csv: messages data to process
	+ process_data.py: ETL script to process data
	+ InsertDatabaseName.db: database to save clean data to
+ models
	+ train_classifier.py: script that runs a machine learning pipeline
	+ classifier.pkl: saved sqlite model
+ .gitignore
+ README.md
+ requirements.txt

## Files desctiption

**process_data.py** - the ETL script which takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.

**train_classifier.py** - The machine learning script which takes the database file path and model file path, creates and trains a multi-output supervised model, and stores the model into a pickle file to the specified model file path.

**run.py** - the script runs the web app which extracts data from the database to provide data visualisations and uses the trained model to classify new messages into 36 categories. 

Additional files:

**ETL Pipeline Preparation.ipynb** - a Python notebook that contains the code for the data preparation.

**ML_Pipeline_Preparation.ipynb** - a Python notebook that contains the code with analysis and testing of a machine learning pipeline with various classifiers. 

## Installation and running

This project requires Python 3.x and the following Python libraries installed:

pandas
scikit-learn
plotly
nltk
sqlalchemy
flask

To install all libraries: pip install -r requirements.txt 

To run the project:

1. Run ETL script:
	
	python process_data.py disaster_messages.csv disaster_categories.csv InsertDatabaseName.db

2. Run a machine learning script:
	
	python train_classifier.py ../data/InsertDatabaseName.db classifier.pkl

3. Run a Flask web app:

	This runs a Flask web app 


## Summary of Findings

Several machine learning algorithms have been tested: 
1. RandomForestClassifier
2. MultinomialNB
3. Logistic Regression

The data was gathered and cleaned using the ETL script.

The machine learning pipeline was created to train multi-output Logistic Regression model into various categories. 

Flask app was created to give the training data overview and classify the messages the user entered. 


## Acknowledgments

This project was done as the part of Udacity learning. The data have been provided by [Appen](https://appen.com/).