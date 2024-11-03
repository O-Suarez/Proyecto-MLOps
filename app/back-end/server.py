import json
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import random

import xgboost as xgb
from modeling import model_training, preprocess, data_loading
import logging
import uvicorn
import matplotlib
from urllib.parse import quote

# Carga los datos
columns = ["NEds", "NActDays", "pagesWomen", "wikiprojWomen"]
X_train, X_test, y_train, y_test = preprocess.get_data(columns)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de MLflow
mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment("wikipedia")
mlflow.sklearn.autolog()
matplotlib.use('Agg')

# Inicializar la app
app = FastAPI()

# Agregar CORS a la app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Definir el esquema para los datos de entrenamiento
class TrainingData(BaseModel):
    classifier_type: str
    hyperparameters: Dict
    NEds: int
    NActDays: int
    pagesWomen: int
    wikiprojWomen: int
    additional_data: List[int]

class ModelData(BaseModel):
    classifier_type: str
    hyperparameters: Dict
    additional_data: List[int] = []

class BestModelRequest(BaseModel):
    experiment_name: str
    metric_name: str = "accuracy"
    classifier_type: str = "logistic_regression"

class HyperparameterSearchData(BaseModel):
    classifier_type: str
    param_distributions: Dict[str, List]
    cv: int = 5
    search_method: str = "grid_search"

# Función para crear un modelo basado en el tipo de clasificador
def create_model(classifier_type, hyperparameters, random_state=42):
    if classifier_type == 'logistic_regression':
        return LogisticRegression(random_state=random_state, **hyperparameters)
    elif classifier_type == 'random_forest':
        return RandomForestClassifier(random_state=random_state, **hyperparameters)
    elif classifier_type == 'svm':
        return SVC(random_state=random_state, **hyperparameters, probability=True)
    elif classifier_type == 'xgboost':
        return xgb.XGBClassifier(random_state=random_state, **hyperparameters)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

# Function to perform hyperparameter search using Grid Search
def hyperparameter_search_generator_grid(classifier_type, param_distributions, cv):
    try:
        param_grid = list(ParameterGrid(param_distributions))
        total_iterations = len(param_grid)
        valid_iterations = 0
        for idx, params in enumerate(param_grid):
            params = {k: (None if v == 'None' else v) for k, v in params.items()}
            accuracy = model_training.train_and_log_model(
                X_train, y_train, X_test, y_test,
                classifier_type=classifier_type,
                hyperparameters=params
            )
            valid_iterations += 1
            result = {
                'iteration': valid_iterations,
                'total_iterations': total_iterations,
                'params': params,
                'score': accuracy
            }
            yield json.dumps(result) + "\n"
    except Exception as e:
        logger.error(f"Error en hyperparameter_search_generator_grid: {str(e)}")
        yield json.dumps({'error': str(e)}) + "\n"

# Function to perform hyperparameter search using Halving Random Search
def hyperparameter_search_generator_halving(classifier_type, param_distributions, cv):
    try:
        from scipy.stats import uniform, randint

        # Define which parameters require integer values for each classifier
        integer_params = {
            'random_forest': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
            'logistic_regression': [],
            'svm': [],
            'xgboost': ['n_estimators', 'max_depth']
        }

        # Convert lists to distributions
        distributions = {}
        for param, values in param_distributions.items():
            # Handle 'None' and string values
            if any(v is None or isinstance(v, str) for v in values):
                distributions[param] = values  # Categorical parameter
            else:
                # Values are numeric
                min_val = min(values)
                max_val = max(values)
                if param in integer_params.get(classifier_type, []):
                    # Parameter requires integer values
                    min_int = int(np.floor(min_val))
                    max_int = int(np.ceil(max_val))
                    if min_int == max_int:
                        distributions[param] = [min_int]
                    else:
                        distributions[param] = randint(min_int, max_int + 1)
                else:
                    # Parameter accepts float values
                    if min_val == max_val:
                        distributions[param] = [min_val]
                    else:
                        distributions[param] = uniform(min_val, max_val - min_val)

        # Create the base estimator
        base_estimator = create_model(classifier_type, {}, random_state=42)

        # Create HalvingRandomSearchCV
        halving_search = HalvingRandomSearchCV(
            estimator=base_estimator,
            param_distributions=distributions,
            cv=cv,
            factor=3,
            random_state=42,
            n_jobs=1,
            scoring='accuracy',
            verbose=1,
            error_score='raise'
        )

        # Perform the search
        halving_search.fit(X_train, y_train)

        # Iterate over the results
        results = halving_search.cv_results_
        for i in range(len(results['params'])):
            params = results['params'][i]
            # Convert integer parameters to integers
            for param in params:
                if param in integer_params.get(classifier_type, []):
                    params[param] = int(params[param])
            score = results['mean_test_score'][i]
            result = {
                'iteration': i + 1,
                'total_iterations': len(results['params']),
                'params': params,
                'score': score
            }
            # Log the model using train_and_log_model for consistency
            model_training.train_and_log_model(
                X_train, y_train, X_test, y_test,
                classifier_type=classifier_type,
                hyperparameters=params
            )
            yield json.dumps(result) + "\n"

    except Exception as e:
        logger.error(f"Error in hyperparameter_search_generator_halving: {str(e)}")
        yield json.dumps({'error': str(e)}) + "\n"

# **Add the missing endpoint here**
@app.post("/hyperparameter_search")
async def hyperparameter_search(data: HyperparameterSearchData, request: Request):
    async def event_generator():
        try:
            if data.search_method == 'halving_random_search':
                generator = hyperparameter_search_generator_halving(
                    data.classifier_type, data.param_distributions, data.cv)
            else:
                generator = hyperparameter_search_generator_grid(
                    data.classifier_type, data.param_distributions, data.cv)
            for event in generator:
                yield event
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected")
                    break
        except Exception as e:
            logger.error(f"Error in hyperparameter_search: {str(e)}")
            yield json.dumps({'error': str(e)}) + "\n"
    return StreamingResponse(event_generator(), media_type="text/plain")

# Función para obtener el mejor modelo basado en una métrica
@app.post("/get_best_model")
def get_best_model_by_type(request: BestModelRequest):
    try:
        client = MlflowClient()

        # Encontrar el experimento por su nombre
        experiment = client.get_experiment_by_name(request.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{request.experiment_name}' not found.")

        # Buscar todos los runs del experimento
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.classifier_type = '{request.classifier_type}'",
            order_by=[f"metrics.{request.metric_name} DESC"],
            max_results=1
        )

        if not runs:
            raise ValueError(f"No runs found for experiment '{request.experiment_name}' with model type '{request.classifier_type}'.")

        best_run = runs[0]

        # Construye el link de MLFlow
        tracking_uri = mlflow.get_tracking_uri()
        experiment_id = experiment.experiment_id
        run_id = best_run.info.run_id

        # Realiza el encoding de los IDs
        experiment_id_encoded = quote(str(experiment_id))
        run_id_encoded = quote(str(run_id))

        mlflow_link = f"{tracking_uri}/#/experiments/{experiment_id_encoded}/runs/{run_id_encoded}"

        return {
            "hyperparameters": best_run.data.params,
            "random_state": best_run.data.params.get("random_state"),
            "metric_value": best_run.data.metrics.get(request.metric_name),
            "mlflow_link": mlflow_link
        }
    except Exception as e:
        logger.error(f"Error finding the model: {str(e)}")
        return {"error": str(e)}

# Función para obtener el mejor modelo basado en una métrica
def get_best_model(experiment_name: str, metric_name: str = "accuracy"):
    try:
        # Crear un cliente de MLflow
        client = MlflowClient()

        # Obtener el ID del experimento por su nombre
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"El experimento '{experiment_name}' no fue encontrado.")

        # Consultar todos los runs del experimento
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],  # Ordenar por la métrica en orden descendente
            max_results=1  # Solo obtener el mejor run
        )

        if not runs:
            raise ValueError(f"No se encontraron runs para el experimento '{experiment_name}' con la métrica '{metric_name}'.")

        # Obtener el mejor run
        best_run = runs[0]
        best_run_id = best_run.info.run_id

        logger.info(f"El mejor run tiene el ID: {best_run_id} con un {metric_name} de: {best_run.data.metrics[metric_name]}")

        # Cargar el modelo correspondiente al mejor run
        model_uri = f"runs:/{best_run_id}/model"
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Error al obtener el mejor modelo: {str(e)}")
        return None

# Ruta para entrenar el modelo
@app.post("/train")
def train_model(data: ModelData):
    try:
        # Preprocesar los datos (aquí puedes transformar los datos según sea necesario)
        #columns = ["NEds", "NActDays", "pagesWomen", "wikiprojWomen"]

        # Convertir las listas a DataFrames
        #X_train, X_test, y_train, y_test = preprocess.get_data(columns)

        # Entrenar el modelo y obtener la precisión
        accuracy = model_training.train_and_log_model(
            X_train, y_train, X_test, y_test,
            classifier_type=data.classifier_type,
            hyperparameters=data.hyperparameters
        )
        return {
            "status": "Model trained and logged successfully",
            "accuracy": accuracy
        }
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return {"error": str(e)}

@app.post('/fetch_test_row')
def fetch_test_row(gender_type: int = Body(default=0, embed=True)):
    # Carga los datos
    columns = ["E_NEds", "NActDays", "pagesWomen", "wikiprojWomen"]
    raw_data = pd.read_csv("modeling/data/data.csv")
    
    # Filtra los datos por género
    if gender_type in [0, 1, 2]:
        filtered_data = raw_data[raw_data['gender'] == gender_type]
    else:
        filtered_data = raw_data
    
    # Verifica si hay datos disponibles
    if filtered_data.empty:
        return JSONResponse(content={"error": "No data available for the specified gender type."}, status_code=400)
    
    # Selección aleatoria de una fila
    sampled_row = filtered_data.sample(n=1).iloc[0]
    
    # Extrae las columnas seleccionadas
    row = sampled_row[columns]
    expected = str(sampled_row['gender'])
    
    # Convierte la fila a un diccionario para la respuesta del frontend
    row_dict = row.to_dict()

    return JSONResponse(content={"row": row_dict, "expected": expected})

# Ruta para predecir usando el mejor modelo
@app.post("/predict")
def predict_model(data: TrainingData):
    try:
        # Obtener el mejor modelo del experimento basado en la métrica 'accuracy'
        model = get_best_model(experiment_name="wikipedia", metric_name="accuracy")
        # Preprocesar los datos para la predicción (ajustar según tus necesidades)
        input_data = pd.DataFrame([{
            "NEds": data.NEds,
            "NActDays": data.NActDays,
            "pagesWomen": data.pagesWomen,
            "wikiprojWomen": data.wikiprojWomen
        }])
        # Realizar la predicción
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

# Ruta GET para verificar que la API está corriendo
@app.get("/api/")
async def get_status():
    return {
        "message": "La API está corriendo"
    }

# Configuración para ejecutar la app
if __name__ == "__main__":
    logger.info("Iniciando servidor...")
    uvicorn.run(
        app,  # Cambié "server:app" a app
        host="0.0.0.0",  # Cambié a "0.0.0.0" para que sea accesible desde fuera del contenedor
        port=8887,        
        log_level="info", # Nivel de logs
        reload=False       # Habilita la recarga automática en desarrollo
    )
