import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Returns a merged dataframe of 2 csv files: messages_filepath and categories_filepath.
    Args:
        messages_filepath: input CSV file with messages
        categories_filepath: input CSV file with categories
    Returns:
        df: dataframe of merged input files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    '''
    Returns a cleaned input dataframe:
        1. Splits the values of "categories" column into individual category columns
        2. Converts category values to number 0 and 1.
        3. Drops duplicates
        4. Drops columns with only one unique value
    Args:
        df: input dataframe
    Returns:
        df: cleaned input dataframe
    '''   
    # values of "categories" column are splitted into individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # rename the columns of `categories`
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # replaces values >1 with 1 for binary output
    categories.loc[categories['related'] > 1, 'related'] = 1
    
    # drop columns with only one unique value
    for col in categories.columns:
        if len(categories[col].unique()) == 1:
            categories.drop(col,inplace=True,axis=1)
            
    # drop the original categories column from "df"       
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new "categories" dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    '''
    Saves dataframe df into an sqlite database: database_filename.
    Args:
        df: dataframe
        database_filename: input CSV file with categories
    Returns:
        df: dataframe of merged input files
    '''    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False)  

def main():
    '''
    1. Loads input data messages_filepath and categories_filepath into one dataframe using load_data() function
    2. Cleans the dataframe using clean_data() function
    3. Saves cleaned dataset into sqlite database using save_data() function
    
    Args:
        df: dataframe
        database_filename: input CSV file with categories
    Returns:
        df: dataframe of merged input files
    '''       
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()