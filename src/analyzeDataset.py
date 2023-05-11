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


def plot_ds2D(X,y):
    sns.scatterplot(x=X[0],y=X[1],hue=y,palette="deep")
    plt.show()

dataset = openml.datasets.get_dataset('credit-g')

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Dataset {dataset.name}')
plt.colorbar(label='Target')
plt.show()