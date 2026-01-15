import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)

    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target"
    ]

    encoder = LabelEncoder()
    df["target"] = encoder.fit_transform(df["target"])

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y
