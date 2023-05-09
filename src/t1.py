from dataset import split_dataset
import openml
import pandas as pd

dataset = openml.datasets.get_dataset('diabetes')

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
df = pd.DataFrame(X, columns=attribute_names)
df["class"] = y

print(df.columns)

dataframes = split_dataset(df, n_features=2, n_df=5)

for i in range(len(dataframes)):
    print(f"Dataframe {i+1}:\n {dataframes[i].head()}")