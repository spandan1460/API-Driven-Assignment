import mlflow
import mlflow.sklearn
import pandas as pd
import psutil
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

mlflow.set_tracking_uri("file:./mlruns")

def prepare_data():
    df = pd.read_csv("data/titanic_processed.csv")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"
    ]
    feature_cols += [c for c in df.columns if c.startswith("Embarked_")]

    X = df[feature_cols]
    y = df["Survived"]
    return X, y

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        end_time = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        confusion = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time_seconds", end_time - start_time)

        mlflow.log_metric("true_positive", int(confusion[1][1]))
        mlflow.log_metric("false_positive", int(confusion[0][1]))
        mlflow.log_metric("true_negative", int(confusion[0][0]))
        mlflow.log_metric("false_negative", int(confusion[1][0]))

        mlflow.log_metric("system_cpu_usage", psutil.cpu_percent(interval=1))
        mlflow.log_metric("system_memory_usage", psutil.virtual_memory().percent)

        if model_name == "RandomForest":
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("n_estimators", model.n_estimators)
        else:
            mlflow.log_param("max_iter", model.max_iter)

        mlflow.sklearn.log_model(model, "model")

        print(f"Successfully logged {model_name} to MLflow.")
        print(f"Accuracy: {accuracy:.4f}")

def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    mlflow.set_experiment("Titanic_Survival_Local")

    models = {
        "RandomForest": RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    for model_name, model in models.items():
        train_and_log_model(model_name, model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()