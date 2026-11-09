# train_model.py
import io
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- Activity A: retrieve the MATHEMATICS file ----------
# UCI dataset 320 ships two files: student-mat.csv (Math) and student-por.csv
# (Portuguese). We read student-mat.csv explicitly so the data is really Math.
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def load_math_csv(url=URL):
    raw = urllib.request.urlopen(url).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    if "student-mat.csv" not in names:  # some mirrors nest a second zip
        inner = next(n for n in names if n.endswith(".zip"))
        zf = zipfile.ZipFile(io.BytesIO(zf.read(inner)))
    with zf.open("student-mat.csv") as f:
        return pd.read_csv(f, sep=";")


df = load_math_csv()
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# ---------- Activity B: preprocess, train, evaluate ----------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor (target leakage)
X = df.drop(columns=["G3"])
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )),
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
