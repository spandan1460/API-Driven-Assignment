import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from prefect import flow, task, get_run_logger

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

@task(name="Data Ingestion")
def ingest_data():
    logger = get_run_logger()
    df = pd.read_csv("data/titanic.csv")
    logger.info(f"Ingested {len(df)} records from local storage.")
    return df

@task(name="Data Pre-processing")
def preprocess(df):
    logger = get_run_logger()
    df = df.copy()

    logger.info(f"Summary Statistics:\n{df.describe(include='all').to_string()}")
    logger.info(f"Missing Values:\n{df.isnull().sum().to_string()}")
    logger.info(f"Data Types:\n{df.dtypes.to_string()}")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

    scaler = MinMaxScaler()
    df[["Age", "Fare", "SibSp", "Parch", "FamilySize"]] = scaler.fit_transform(
        df[["Age", "Fare", "SibSp", "Parch", "FamilySize"]]
    )

    df.to_csv("data/titanic_processed.csv", index=False)
    logger.info("Saved processed dataset to data/titanic_processed.csv")
    return df

@task(name="Exploratory Data Analysis")
def run_eda(df):
    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    df["Survived"].value_counts().plot(kind="bar")
    plt.title("Survival Distribution")
    plt.tight_layout()
    plt.savefig("plots/survival_dist.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Sex", hue="Survived")
    plt.title("Survival by Sex")
    plt.tight_layout()
    plt.savefig("plots/survival_by_sex.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Pclass", hue="Survived")
    plt.title("Survival by Passenger Class")
    plt.tight_layout()
    plt.savefig("plots/survival_by_pclass.png")
    plt.close()

    return "EDA charts saved to plots/"

@flow(name="Titanic-DataOps-Local")
def data_ops_flow():
    raw_data = ingest_data()
    clean_data = preprocess(raw_data)
    run_eda(clean_data)

if __name__ == "__main__":
    data_ops_flow.serve(name="titanic-local-deployment", cron="*/3 * * * *")