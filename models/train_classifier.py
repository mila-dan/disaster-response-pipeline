import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
    
def load_data(database_filepath):
    '''
    Loads dataset from sqlite database database_filepath and splits data into features X and targets y.
    Args:
        database_filepath - a path to database to load the data
    Returns:
        X - array of features (text messages)
        y - array of targets (0 or 1)
        categories - list of target names
    '''
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath.split('/')[-1], con=engine)

    X = df['message'].values
    y = df[df.columns[4:]].values
    categories = list(df.columns[4:])
    return X, y, categories 


def tokenize(text):
    '''
    Splits a text input into a list of filtered, stemmed and lemmatized tokens. 
    
    Args:
        text - input string
    Returns:
        tokens - a list of tokens
    '''  
    import re

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.stem.porter import PorterStemmer

    url_regex = '[](?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+' 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    text = re.sub(r"[^a-zA-z0-9]", " ", text.lower())
    #remove brackets
    text = re.sub(r"[\([{})\]]", "", text)
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english") and not word.isdigit()]

    tokens = [PorterStemmer().stem(word) for word in tokens]
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    
    return tokens


def build_model():
    '''
    Builds a machine learning pipeline:
        1. CountVectorizer() - converts a collection of text documents into a matrix of token counts using tokenize() function 
        2. TfidfTransformer() - transforms a matrix of token countsinto a matrix of TF-IDF scores
        3. MultiOutputClassifier() - multi-output classification using input classifier
    Args:
        classifier - a machine learning classifier
    Returns:
        pipeline - a classification result
    '''  
    from sklearn.pipeline import Pipeline
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer    
    from warnings import simplefilter

    # ignore all warnings
    simplefilter(action='ignore')
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.85)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(LogisticRegression(penalty='l1',solver='saga', C=10.0, max_iter=200)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints classification report
    ''' 
    #return model.score(X_test, Y_test)

    Y_pred = model.predict(X_test)
    return classification_report(Y_test, Y_pred, target_names=category_names, zero_division=0)

def save_model(model, model_filepath):
    '''
    Saves the model into model_filepath pickle file
    
    Args:
        model - a machine learning model
        model_filepath - a model file name and path
    '''  
    import pickle
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    '''
    This function:
        1. imports data from database using load_data() function.
        2. splits data into train and test
        3. builds a machine learning model using build_model() function.
        4. fit the model with training data 
        5. evaluates model
        6. saves model using save_model()
    ''' 

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        report = evaluate_model(model, X_test, Y_test, category_names)
        print(report)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()