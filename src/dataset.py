import pandas as pd
import random


def split_dataset(df: pd.DataFrame, n_features=None) -> list:
    """ Split dataset into different features.
        If n_features is not specified then the dataset will be split into 2 or 3 features
        according to whether it has an even or odd number of features, respectively."""
    dataframes = []  # list of dataframes to return
    cols = [i for i in df.columns]
    if n_features == None:
        if len(cols) % 2 != 0:
            n_features = 3
        else:
            n_features = 2
    Y_class = cols[-1]  # assumes last column is the class
    cols.remove(Y_class)

    '''Until cols is empty, select n_features columns at random and create a dataframe with them, adding it to the list'''
    
    l = []
    while len(cols) != 0:
        for i in range(n_features):
            if len(cols) != 0:
                c = random.choice(cols)
                cols.remove(c)
                l += [c]
        dataframes.append(df[l + [Y_class]])
        l = []
    return dataframes
