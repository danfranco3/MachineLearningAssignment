from dataset import split_dataset
import openml
import pandas as pd
from svm import SVM
from sklearn.model_selection import train_test_split
import numpy as np


def sign(value):
    if value >= 0:
        return 1
    else:
        return -1

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

dataset = openml.datasets.get_dataset('diabetes')

X1, y1, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
df = pd.DataFrame(X1, columns=attribute_names)
df["class"] = y1

dataframes = split_dataset(df, n_features=2, n_df=28)

X = []
y = df["class"]
print(y1)
print(y)

for i in range(len(dataframes)):
    #print(f"Dataframe {i+1}:\n {dataframes[i].head()}")
    X.append(dataframes[i].iloc[0:dataframes[i].shape[0],0:dataframes[i].shape[1] - 1])

accs = []

svmPredictions = [None] * len(dataframes)         # set of predictions from each svm
resultsSize = 0                                   # nr of predictions

for i in range(len(dataframes)):
    XF = X[i].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(XF, y, test_size=0.2, shuffle=True, random_state=1)
    
    clf = SVM(n_iters=1000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions)
    
    prediction = np.sign(predictions)
    np.where(prediction == -1, 0, 1)
    
    
    
    acc = accuracy(y_test, prediction)
    accs.append(acc)
    print("SVM Accuracy: ", acc)

print("Average Accuracy: ", sum(accs)/len(accs))

nrOfSVMs = len(svmPredictions)
resultsSize = len(svmPredictions[0])
finalPredictions = [0] * resultsSize

# calculating the weights
for i in range(resultsSize):

    absValPred = [0] * nrOfSVMs             # distance to the hyperplane
    rank = [0] * nrOfSVMs                   # ranking of the most certain svm's
    pred = [0] * nrOfSVMs                   # prediction from each SVM
    sumTotal = 0                            # factor to calculate the weights


    for j in range(nrOfSVMs):
        absValPred[j] = abs(svmPredictions[j][i])
        pred[j] = sign(svmPredictions[j][i])
        sumTotal += (j+1)

    for j in range(nrOfSVMs):
        rank[j] = absValPred.index(max(absValPred))               # ranks from best to worst
        absValPred[rank[j]] = 0

    for j in range(nrOfSVMs):
        weigth = (nrOfSVMs - j) / sumTotal
        finalPredictions[i] += (weigth * pred[rank[j]])

    if finalPredictions[i] >= 0:
        finalPredictions[i] = 1
    else:
        finalPredictions[i] = 0


print(finalPredictions)
print("SVM Accuracy: ", accuracy(y_test, finalPredictions))