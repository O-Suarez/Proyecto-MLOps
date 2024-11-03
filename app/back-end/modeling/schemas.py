from pydantic import BaseModel
from typing import List, Dict

class TrainingData(BaseModel):
    X_train: List[List[float]]
    y_train: List[int]
    X_test: List[List[float]]
    y_test: List[int]
    classifier_type: str = 'logistic_regression'
    hyperparameters: Dict = {}  


class PredictionData(BaseModel):
    NEds: int
    NActDays: int
    pagesWomen: int
    wikiprojWomen: int