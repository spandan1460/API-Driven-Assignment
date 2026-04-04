from fastapi import FastAPI
import mlflow
import mlflow.tracking

app = FastAPI(title="Titanic Local ML Application API")
client = mlflow.tracking.MlflowClient()

PREFECT_API_KEY = os.getenv("PREFECT_API_KEY")
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "https://api.prefect.cloud/graphql")
PREFECT_FLOW_NAME = os.getenv("PREFECT_FLOW_NAME", "Titanic-DataOps-Local")

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
    if not PREFECT_API_KEY:
        return {"error": "PREFECT_API_KEY environment variable is not set."}

    headers = {
        "Authorization": f"Bearer {PREFECT_API_KEY}",
        "Content-Type": "application/json",
    }

    # Query flows + last N flow runs for the named flow
    query = """
    query ($flow_name: String!, $limit: Int!) {
      flows(where: { name: { _eq: $flow_name } }) {
        id
        name
      }
      flow_runs(where: { flow: { name: { _eq: $flow_name } } }, order_by: { start_time: desc }, limit: $limit) {
        id
        start_time
        end_time
        state
      }
    }
    """
    variables = {"flow_name": PREFECT_FLOW_NAME, "limit": 10}

    resp = requests.post(PREFECT_API_URL, json={"query": query, "variables": variables}, headers=headers, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data", {})
    runs = data.get("flow_runs", [])

    if not runs:
        return {
            "flow_name": PREFECT_FLOW_NAME,
            "Status": "No runs found",
            "DataOps_Frequency": "Unknown",
            "run_count_returned": 0
        }

    def parse_ts(ts):
        if not ts:
            return None
        # ISO8601 with Z -> +00:00 for fromisoformat
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    last = runs[0]
    last_start = parse_ts(last.get("start_time"))
    last_end = parse_ts(last.get("end_time"))

    # run counts (returned window) and overall estimate
    run_count = len(runs)
    starts = [parse_ts(r.get("start_time")) for r in runs if r.get("start_time")]
    freq_minutes = None
    if len(starts) >= 2:
        deltas = []
        for i in range(len(starts) - 1):
            dt = (starts[i] - starts[i + 1]).total_seconds() / 60.0
            if dt > 0:
                deltas.append(dt)
        if deltas:
            freq_minutes = round(statistics.mean(deltas), 2)

    return {
        "flow_name": PREFECT_FLOW_NAME,
        "last_run_status": last.get("state"),
        "last_run_start": last_start.isoformat() if last_start else None,
        "last_run_end": last_end.isoformat() if last_end else None,
        "run_count_returned": run_count,
        "estimated_run_frequency_minutes": freq_minutes
    }