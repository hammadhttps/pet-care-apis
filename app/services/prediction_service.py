import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List
from pathlib import Path

class PredictionService:
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.cat_model = None
        self.dog_model = None
        self.cat_features = None
        self.dog_features = None
        self.cat_label_encoder = None
        self.dog_label_encoder = None
        self.disease_categories = None
        
    def load_models(self):
        """Load both cat and dog models"""
        cat_model_path = os.path.join(self.models_path, "cat_model.pkl")
        dog_model_path = os.path.join(self.models_path, "dog_model.pkl")
        
        # Load cat model
        if os.path.exists(cat_model_path):
            model_data = joblib.load(cat_model_path)
            if isinstance(model_data, dict):
                self.cat_model = model_data.get('model')
                self.cat_features = model_data.get('features') or model_data.get('feature_columns')
                if isinstance(self.cat_features, dict):
                    ordered_groups = ["categorical", "numerical", "binary"]
                    flattened: List[str] = []
                    for group in ordered_groups:
                        cols = self.cat_features.get(group) or []
                        flattened.extend(list(cols))
                    self.cat_features = flattened

                self.cat_label_encoder = model_data.get('label_encoder')
                self.disease_categories = model_data.get('categories') or model_data.get('disease_names')
            else:
                self.cat_model = model_data
        
        # Load dog model
        if os.path.exists(dog_model_path):
            model_data = joblib.load(dog_model_path)
            if isinstance(model_data, dict):
                self.dog_model = model_data.get('model')
                self.dog_features = model_data.get('features') or model_data.get('feature_columns')
                if isinstance(self.dog_features, dict):
                    ordered_groups = ["categorical", "numerical", "binary"]
                    flattened: List[str] = []
                    for group in ordered_groups:
                        cols = self.dog_features.get(group) or []
                        flattened.extend(list(cols))
                    self.dog_features = flattened

                self.dog_label_encoder = model_data.get('label_encoder')
                self.disease_categories = model_data.get('categories') or model_data.get('disease_names')
            else:
                self.dog_model = model_data
    
    def preprocess_data(self, animal_type: str, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction.

        The saved pipelines already contain their own preprocessing (encoders,
        scalers, etc.). We therefore keep the raw feature columns intact and
        simply ensure that every expected column exists and is ordered
        correctly. This avoids double‑encoding and the missing-column errors
        we were seeing.
        """
        df = pd.DataFrame([data])

        base_features = [
            "Sex",
            "Breed",
            "Age",
            "Weight",
            "Symptom_1",
            "Symptom_2",
            "Symptom_3",
            "Symptom_4",
            "Appetite_Loss",
            "Vomiting",
            "Diarrhea",
            "Coughing",
            "Labored_Breathing",
            "Body_Temperature_in_Celsius",
        ]

        for feature in base_features:
            if feature not in df.columns:
                df[feature] = np.nan

        # Use stored feature ordering when available, otherwise fall back to defaults.
        if animal_type == "cat" and self.cat_features:
            ordered = self.cat_features
        elif animal_type == "dog" and self.dog_features:
            ordered = self.dog_features
        else:
            ordered = base_features

        # Ensure only expected columns are passed to the model and in correct order.
        df_ordered = df[ordered].copy()

        return df_ordered
    
    def predict(self, animal_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using appropriate model"""
        
        # Select model
        model = self.cat_model if animal_type == 'cat' else self.dog_model
        label_encoder = self.cat_label_encoder if animal_type == "cat" else self.dog_label_encoder
        
        if model is None:
            raise ValueError(f"{animal_type.capitalize()} model not loaded")

        # Avoid multiprocessing errors in restricted environments by forcing single-threaded predict
        try:
            n_job_params = {
                key: 1
                for key, value in model.get_params().items()
                if key.endswith("n_jobs") and value not in (None, 1)
            }
            if n_job_params:
                model.set_params(**n_job_params)
        except Exception:
            # Best-effort; proceed even if get_params is not available
            pass
        
        # Preprocess data
        processed_data = self.preprocess_data(animal_type, data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get prediction probabilities if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)[0]
            confidence = float(max(probabilities))
        
        # Convert encoded numeric prediction to disease label when possible
        if label_encoder is not None and isinstance(prediction, (int, np.integer)):
            try:
                prediction = label_encoder.inverse_transform([int(prediction)])[0]
            except Exception:
                pass

        # Back-compat: support dict/list mappings if present
        if self.disease_categories and isinstance(prediction, (int, np.integer)):
            if isinstance(self.disease_categories, dict):
                prediction = self.disease_categories.get(prediction, prediction)
            elif isinstance(self.disease_categories, (list, tuple, np.ndarray)):
                idx = int(prediction)
                if 0 <= idx < len(self.disease_categories):
                    prediction = self.disease_categories[idx]
        
        return {
            'prediction': str(prediction),
            'confidence': confidence
        }
    
    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded models"""
        models_info = []
        
        if self.cat_model:
            models_info.append({
                'animal_type': 'cat',
                'is_loaded': True,
                'num_features': len(self.cat_features) if self.cat_features else None,
                'model_type': type(self.cat_model).__name__
            })
        else:
            models_info.append({
                'animal_type': 'cat',
                'is_loaded': False
            })
        
        if self.dog_model:
            models_info.append({
                'animal_type': 'dog',
                'is_loaded': True,
                'num_features': len(self.dog_features) if self.dog_features else None,
                'model_type': type(self.dog_model).__name__
            })
        else:
            models_info.append({
                'animal_type': 'dog',
                'is_loaded': False
            })
        
        return models_info
