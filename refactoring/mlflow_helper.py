import subprocess
import time
import requests
import mlflow
from mlflow.tracking import MlflowClient

docker_desktop_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"

def load_last_mlflow_model(tracking_uri, experiment_name):
    # Set the tracking URI and experiment name
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Initialize the MLflow client
    client = MlflowClient()

    # Get the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        # Search for the latest run in the experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],  # Order runs by start time in descending order
            max_results=1  # Get only the most recent run
        )

        if runs:
            # Get the run ID of the latest run
            last_run = runs[0]
            run_id = last_run.info.run_id
            print(f"Latest run ID: {run_id}")

            # Construct the model URI using the run ID
            logged_model = f"runs:/{run_id}/model"

            # Load the model from the run
            loaded_model = mlflow.pyfunc.load_model(logged_model)

            return loaded_model, last_run
        else:
            print(f"No runs found in experiment '{experiment_name}'.")
            return None
    else:
        print(f"Experiment '{experiment_name}' not found.")
        return None

def is_docker_running():
    try:
        output = subprocess.check_output(['docker', 'info'], stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        # Docker command not found
        return False

def start_docker_desktop():
    if not is_docker_running():
        print("Starting Docker Desktop...")
        subprocess.Popen(docker_desktop_path)
        # Wait for Docker to start
        while not is_docker_running():
            print("Waiting for Docker to start...")
            time.sleep(5)
        print("Docker is running.")
    else:
        print("Docker Desktop is already running.")

def is_mlflow_running(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except:
        return False

def start_docker_compose():
    subprocess.run(["docker-compose", "up", "-d"], check=True)

def start_mlflow():
    start_docker_desktop()
    time.sleep(5)
    MLFLOW_URL = "http://localhost:5000"
    
    if not is_mlflow_running(MLFLOW_URL):
        print("MLflow is not running. Starting MLflow using docker-compose...")
        start_docker_compose()
        # Wait for MLflow to start
        for _ in range(10):
            if is_mlflow_running(MLFLOW_URL):
                print("MLflow started successfully.")
                break
            else:
                print("Waiting for MLflow to start...")
                time.sleep(6.5)
        else:
            print("Failed to start MLflow.")
    else:
        print("MLflow is already running.")

