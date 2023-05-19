from sklearn.metrics import accuracy_score
from run_svms import test_with_dataset, get_dataset, gen_df
from sklearn.model_selection import train_test_split
from sklearn import svm

# dataset_names = ['wdbc', 'diabetes', 'ilpd', 'electricity', 'bank-marketing']

dataset_names = ['diabetes']

TEST_SIZE = 0.2

parameter_variation = [(28, 2, False), (28, 2, False), (26, 2, True), (27, 3, True)]

for name in dataset_names:
    for (n_df, n_features, majority) in parameter_variation:
        print(f"Testing with {name} dataset, {n_df} sub-datasets and {n_features} features")
        df = test_with_dataset(name, n_df, n_features, random_state=1, test_size=TEST_SIZE, majority=majority)
        df.to_csv(f"../results/{name}_{n_df}_{n_features}.csv")

    dataset = get_dataset(name)

    print(f"Testing with {name} dataset, scikit-learn SVM")

    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=TEST_SIZE, shuffle=True, random_state=1)

    clf = svm.SVC(kernel='linear')

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    df = gen_df(name, -1, -1, acc)

    df.to_csv(f"../results/{name}_scikit_svm.csv")