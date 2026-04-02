from fastapi import FastAPI
import mlflow
import mlflow.tracking

app = FastAPI(title="Titanic Local ML Application API")
client = mlflow.tracking.MlflowClient()

@app.get("/application-details")
def get_details():
    experiment = client.get_experiment_by_name("Titanic_Survival_Local")
    return {
        "Application_Objective": "Titanic Survival Prediction",
        "Storage_Type": "EC2 Local File System",
        "MLflow_Experiment_ID": experiment.experiment_id if experiment else "Not Initialized",
        "MLflow_Tracking_URI": mlflow.get_tracking_uri(),
        "Artifact_Location": experiment.artifact_location if experiment else "N/A"
    }

@app.get("/pipeline-status")
def get_status():
    return {
        "Status": "Operational",
        "DataOps_Frequency": "Every 3 Minutes"
    }