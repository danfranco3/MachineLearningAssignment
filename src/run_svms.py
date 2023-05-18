import openml
from dataset import split_dataset
import openml
import pandas as pd
from svm import SVM
from sklearn.model_selection import train_test_split
import numpy as np
from weights import calc_weights

def get_dataset(dataset_name):
    dataset = openml.datasets.get_dataset(dataset_name)

    X1, y1, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X1, columns=attribute_names)
    df["class"] = y1

    return df

def accuracy(y_true, y_pred):
    # print(y_true)
    # print(y_pred)
    return np.sum(y_true==y_pred) / len(y_true)

def test_with_dataset(dataset_name, n_df=28, n_features=2, random_state=1, test_size=0.2, n_iters=1000, majority=False):
    
    df = get_dataset(dataset_name)

    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=1)

    subdataframe_list = split_dataset(train, 2, 28, randomize_list=False)

    svmPredictions = [None] * len(subdataframe_list)         # set of predictions from each svm
    resultsSize = 0                                   # nr of predictions

    for i in range(len(subdataframe_list)):
        
        cur_df = subdataframe_list[i]
        clf = SVM(n_iters=1000)
        cols = [c for c in cur_df.columns if c != "class"]
        clf.fit(cur_df[cols].to_numpy(), cur_df["class"].to_numpy())
        predictions = clf.predict(test[cols].to_numpy())
        svmPredictions[i] = predictions

        prediction = np.sign(predictions)
        np.where(prediction == -1, 0, 1)

        print("SVM Accuracy: ", accuracy(test["class"].to_numpy(), prediction))

    print("Calculating weights...")

    finalPredictions = calc_weights(svmPredictions, resultsSize, majority)

    print(test)

    y_pred = test["class"].to_numpy()

    print("Final SVM Accuracy: ", accuracy(y_pred, finalPredictions))

    return gen_df(dataset_name, n_df, accuracy(y_pred, finalPredictions))

def gen_df(dataset_name, n_df, final_acc):
    df = pd.DataFrame.from_dict({'Dataset': dataset_name, 'n_df': n_df, 'final_acc': final_acc}, orient='index').transpose()
    return df

