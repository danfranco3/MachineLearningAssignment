import openml
import numpy as np
import pandas as pd
from run_svms import accuracy, precision, recall, fMeasure
from sklearn.model_selection import train_test_split
from svm import SVM


def singleSVM(name, size):
    dataset = openml.datasets.get_dataset(name)

    X1, y1, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X1, columns=attribute_names)
    df["class"] = y1
    train, test = train_test_split(df, test_size=size, shuffle=True, random_state=1)

    clf = SVM(n_iters=1000)
    cols = [c for c in df.columns if c != "class"]
    clf.fit(df[cols].to_numpy(), df["class"].to_numpy())
    predictions = clf.predict(test[cols].to_numpy())

    prediction = np.sign(predictions)
    prediction = np.where(prediction == -1, 0, 1)

    print("Accuracy of base SVM: ", accuracy(test["class"].to_numpy(), prediction))
    print("Precision of base SVM: ", precision(test["class"].to_numpy(), prediction))
    print("Recall of base SVM: ", recall(test["class"].to_numpy(), prediction))
    print("f-Measure of base SVM: ", fMeasure(test["class"].to_numpy(), prediction))