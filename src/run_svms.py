from dataset import split_dataset
import openml
import pandas as pd
from svm import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
from weights import calc_weights
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns


def get_dataset(dataset_name):
    dataset = openml.datasets.get_dataset(dataset_name)

    X1, y1, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X1, columns=attribute_names)
    df["class"] = y1

    return df

def accuracy(y_true, y_pred):
    return np.sum(y_true==y_pred) / len(y_true)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def fMeasure(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

def confusion_matrix_plot(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def test_with_dataset(dataset_name, n_df=28, n_features=2, random_state=1, size=0.2, n_iters=1000, majority=False, randomize_list = True):
    
    df = get_dataset(dataset_name)

    train, test = train_test_split(df, test_size=size, shuffle=True, random_state=random_state)

    subdataframe_list = split_dataset(train, n_features, n_df, randomize_list=randomize_list)

    svmPredictions = [None] * len(subdataframe_list)         # set of predictions from each svm

    print("Subdataframe list length:", len(subdataframe_list))

    for i in range(len(subdataframe_list)):
        
        cur_df = subdataframe_list[i]
        clf = SVM(n_iters=1000)
        cols = [c for c in cur_df.columns if c != "class"]
        clf.fit(cur_df[cols].to_numpy(), cur_df["class"].to_numpy())
        predictions = clf.predict(test[cols].to_numpy())
        svmPredictions[i] = predictions

        prediction = np.sign(predictions)
        prediction = np.where(prediction == -1, 0, 1)

        print("SVM Accuracy: ", accuracy(test["class"].to_numpy(), prediction))

    print("Calculating weights...")

    resultsSize = len(svmPredictions[0])      # nr of predictions

    finalPredictions = calc_weights(svmPredictions, resultsSize, True)

    y_pred = test["class"].to_numpy()
    finalPredictionsNP = np.array(finalPredictions)

    print("Final SVM Accuracy without weights: ", accuracy(y_pred, finalPredictionsNP))
    print("Final SVM Precision without weights: ", precision(y_pred, finalPredictionsNP))
    print("Final SVM Recall without weights: ", recall(y_pred, finalPredictionsNP))
    print("Final SVM f-Measure without weights: ", fMeasure(y_pred, finalPredictionsNP))
    confusion_matrix_plot(y_pred, finalPredictionsNP)

    finalPredictions = calc_weights(svmPredictions, resultsSize, False)

    finalPredictionsNP = np.array(finalPredictions)

    print("Final SVM Accuracy with weights: ", accuracy(y_pred, finalPredictionsNP))
    print("Final SVM Precision with weights: ", precision(y_pred, finalPredictionsNP))
    print("Final SVM Recall with weights: ", recall(y_pred, finalPredictionsNP))
    print("Final SVM f-Measure with weights: ", fMeasure(y_pred, finalPredictionsNP))
    confusion_matrix_plot(y_pred, finalPredictionsNP)

    return gen_df(dataset_name, n_df, accuracy(y_pred, finalPredictionsNP))

def gen_df(dataset_name, n_df, final_acc):
    df = pd.DataFrame.from_dict({'Dataset': dataset_name, 'n_df': n_df, 'final_acc': final_acc}, orient='index').transpose()
    return df

