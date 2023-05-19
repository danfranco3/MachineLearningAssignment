from sklearn.metrics import accuracy_score
from run_svms import test_with_dataset, get_dataset, gen_df
from sklearn.model_selection import train_test_split
from svm import SVM
import numpy as np
import openml
import pandas as pd
from run_svms import accuracy
from singleSVM import singleSVM

# dataset_names = ['wdbc', 'diabetes', 'ilpd', 'electricity', 'bank-marketing']
def run():
    dataset_names = ['wdbc', 'diabetes', 'ilpd', 'bank-authentication', 'credit-g']

    TEST_SIZE = 0.3

    parameter_variation = [(3, 2, True), (4, 3, True)]

    # [(28, 2, True), (28, 2, False), (26, 2, True), (27, 3, True)]

    for name in dataset_names:

        print(f"Testing {name} dataset.")

        singleSVM(name, TEST_SIZE)

        for (n_df, n_features, majority) in parameter_variation:
            print(f"Testing with {name} dataset, {n_df} sub-datasets and {n_features} features")
            df = test_with_dataset(name, n_df, n_features, random_state=1, size=TEST_SIZE, majority=majority)
            

run()

