import os
import pickle

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "housing.csv")
MODELS_DIR = os.path.join(ROOT, "models")
URL = "https://breathecode.herokuapp.com/asset/internal-link?id=439&path=housing.csv"
FEATURES = ["Latitude", "Longitude", "MedInc"]


def load_housing_data() -> pd.DataFrame:
    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH)
    else:
        df = pd.read_csv(URL)
        os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
        df.to_csv(RAW_PATH, index=False)

    return df[FEATURES]


def main() -> None:
    df = load_housing_data()
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
    kmeans.fit(train_set)

    train_set = train_set.copy()
    train_set["cluster"] = kmeans.labels_.astype(str)

    test_set = test_set.copy()
    test_set["cluster"] = kmeans.predict(test_set[FEATURES]).astype(str)

    X_train = train_set[FEATURES]
    y_train = train_set["cluster"]
    X_test = test_set[FEATURES]
    y_test = test_set["cluster"]

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODELS_DIR, "classifier_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    print(f"Saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
