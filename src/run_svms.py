import openml
from dataset import split_dataset
import openml
import pandas as pd
from svm import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from weights import calc_weights
import matplotlib.pyplot as plt
from itertools import cycle

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
    #print(np.sum(y_true==y_pred))
    #print(len(y_true))
    #print(np.sum(y_true==y_pred) / len(y_true))
    return np.sum(y_true==y_pred) / len(y_true)

def test_with_dataset(dataset_name, n_df=28, n_features=2, random_state=1, size=0.2, n_iters=1000, majority=False):
    
    df = get_dataset(dataset_name)

    train, test = train_test_split(df, test_size=size, shuffle=True, random_state=1)

    subdataframe_list = split_dataset(train, 2, 28, randomize_list=False)

    svmPredictions = [None] * len(subdataframe_list)         # set of predictions from each svm


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

    finalPredictions = calc_weights(svmPredictions, resultsSize, majority)

    y_pred = test["class"].to_numpy()
    finalPredictionsNP = np.array(finalPredictions)

    print("Final SVM Accuracy: ", accuracy(y_pred, finalPredictionsNP))

    roc_curve_calculator(y_pred, finalPredictionsNP)

    return gen_df(dataset_name, n_df, accuracy(y_pred, finalPredictionsNP))

def gen_df(dataset_name, n_df, final_acc):
    df = pd.DataFrame.from_dict({'Dataset': dataset_name, 'n_df': n_df, 'final_acc': final_acc}, orient='index').transpose()
    return df

def roc_curve_calculator(y_test, y_score):

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
