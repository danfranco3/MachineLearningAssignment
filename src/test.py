from dataset import split_dataset
import openml
import pandas as pd
from svm import SVM
from sklearn.model_selection import train_test_split
import numpy as np

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

for i in range(len(dataframes)):
    XF = X[i].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(XF, y, test_size=0.2, shuffle=True, random_state=1)
    
    clf = SVM(n_iters=1000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions)
    acc = accuracy(y_test, predictions)
    accs.append(acc)
    print("SVM Accuracy: ", acc)

print("Average Accuracy: ", sum(accs)/len(accs))


    