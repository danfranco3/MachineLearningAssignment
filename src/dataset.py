import pandas as pd
import random
from itertools import combinations


def split_dataset_wo_repetitions(df: pd.DataFrame, n_features=None) -> list:
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

def split_dataset(df: pd.DataFrame, n_features=None, n_df=20, randomize_list=True) -> list:
    """ Split dataset into different features, select n sub-dataframes and return.
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

    '''Select n_features columns at random and create a dataframe with them, adding it to the list and generating all the possible combinations of sub-dataframes'''
    combinations_wo_repetitions = []
    all_combinations = combinations(cols, r=n_features)
    for tp in all_combinations:
        l=[]
        for i in tp:
            l += [i]
        for t in combinations(tp, r=n_features):
            if t not in combinations_wo_repetitions:
                combinations_wo_repetitions += [t]
                dataframes.append(df[l + [Y_class]])
            else:
                continue

    print(combinations_wo_repetitions)

    if randomize_list:
        random.shuffle(dataframes)

    if (n_df > len(dataframes)) or (n_df == -1):
        return dataframes
    # print(f'Length comb_wo_rept: {len(combinations_wo_repetitions)}')
    # print(f'Length cols: {len(cols)}')
    # print(combinations_wo_repetitions)
    return dataframes[0:n_df]
