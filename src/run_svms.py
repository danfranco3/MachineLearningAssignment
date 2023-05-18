import openml
from dataset import split_dataset
import openml
import pandas as pd
from svm import SVM
from sklearn.model_selection import train_test_split
import numpy as np
from weights import calc_weights
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score, recall_score

def get_dataset(dataset_name):
    dataset = openml.datasets.get_dataset(dataset_name)

    X1, y1, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X1, columns=attribute_names)
    df["class"] = y1

    return df

def test_with_dataset(dataset_name, n_df=28, n_features=2, random_state=1, test_size=0.2, n_iters=1000):
    
    df = get_dataset(dataset_name)

    dataframes = split_dataset(df, n_features, n_df)

    X = []
    y = df["class"]
    # print(y1)
    # print(y)

    for i in range(len(dataframes)):
        #print(f"Dataframe {i+1}:\n {dataframes[i].head()}")
        X.append(dataframes[i].iloc[0:dataframes[i].shape[0],0:dataframes[i].shape[1] - 1])

    accs = []

    svmPredictions = [None] * len(dataframes)         # set of predictions from each svm
    resultsSize = 0                                   # nr of predictions

    print("Calculating SVM predictions...")

    for i in range(len(dataframes)):
        XF = X[i].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(XF, y, test_size=test_size, random_state=random_state)
        
        clf = SVM(n_iters=n_iters)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        # print(predictions)
        svmPredictions[i] = predictions
        
        prediction = np.sign(predictions)
        np.where(prediction == -1, 0, 1)
        
        acc = accuracy(y_test, prediction)
        accs.append(acc)
        print("SVM Accuracy: ", acc)

    avg_acc = sum(accs)/len(accs)

    print("Average Accuracy: ", avg_acc)

    print("Calculating weights...")

    finalPredictions = calc_weights(svmPredictions, resultsSize)

    final_acc = accuracy(svmPredictions, finalPredictions)

    # print(finalPredictions)
    print("Final SVM Accuracy: ", final_acc)

    return gen_df(dataset_name, n_df, avg_acc, final_acc, precision_score(y_test, finalPredictions), recall_score(y_test, finalPredictions))

def gen_df(dataset_name, n_df, avg_acc, final_acc, precision, recall):
    df = pd.DataFrame.from_dict({'Dataset': dataset_name, 'n_df': n_df, 'avg_acc': avg_acc, 'final_acc': final_acc, 'precision': precision, 'recall': recall}, orient='index').transpose()
    return df

