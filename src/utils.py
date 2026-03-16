import os
import sys
import json

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def save_metadata(file_path, model_name, score, all_models=None):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        metadata = {
            "best_model_name": model_name,
            "best_model_score": float(score)
        }
        
        # Add all model scores if provided
        if all_models:
            metadata["all_models"] = {k: float(v) for k, v in all_models.items()}
        
        with open(file_path, "w") as file_obj:
            json.dump(metadata, file_obj, indent=4)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_metadata(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as file_obj:
                return json.load(file_obj)
        return {"best_model_name": "Unknown", "best_model_score": 0.0, "all_models": {}}
    
    except Exception as e:
        raise CustomException(e, sys)
