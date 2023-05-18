import pandas as pd
import numpy as np
import openml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

def holdout_estimation(X,y,models,ts=0.3,seed=0):
    X_train, X_test, y_train, y_test = \
        train_test_split(X,y,test_size=ts,random_state=seed)
    for m in models:
        m.fit(X_train,y_train)
        y_pred = m.predict(X_test)
        print(f'{m} Accuracy:{accuracy_score(y_test,y_pred)}')
        plot_decisionBound(m,X_train,y_train)


def plot_decisionBound(model,X,y):
    disp = DecisionBoundaryDisplay.from_estimator(
        model, X, response_method="predict",
        alpha=0.5)
    disp.ax_.scatter(X['X1'], X['X2'], c=y, edgecolor="k")
    plt.show()


def plot_ds2D(X,y):
    sns.scatterplot(x=X[0],y=X[1],hue=y,palette="deep")
    plt.show()

dataset = openml.datasets.get_dataset('diabetes')

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

linear_svc = SVC(kernel="linear")


print(X.shape)

#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.title(f'Dataset {dataset.name}')
#plt.colorbar(label='Target')
#plt.show()