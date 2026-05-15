import sys
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Optional

from XGBoostModel import OptimizedDiseaseTrainer, VeterinaryFeatureEngineer

# Register classes on __main__ so joblib can unpickle models
# that were saved from a training script run as __main__
sys.modules["__main__"].OptimizedDiseaseTrainer = OptimizedDiseaseTrainer
sys.modules["__main__"].VeterinaryFeatureEngineer = VeterinaryFeatureEngineer


class PredictionService:
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.cat_model: Optional[Any] = None
        self.dog_model: Optional[Any] = None

    def load_models(self):
        cat_model_path = os.path.join(
            self.models_path, "optimized_cat_disease_model.pkl"
        )
        dog_model_path = os.path.join(
            self.models_path, "optimized_dog_disease_model.pkl"
        )

        if os.path.exists(cat_model_path):
            self.cat_model = joblib.load(cat_model_path)

        if os.path.exists(dog_model_path):
            self.dog_model = joblib.load(dog_model_path)

    def predict(
        self, animal_type: str, data: Dict[str, Any], top_n: int = 5
    ) -> Dict[str, Any]:
        model = self.cat_model if animal_type == "cat" else self.dog_model
        if model is None:
            raise ValueError(f"{animal_type.capitalize()} model not loaded")

        df = pd.DataFrame([data])
        X_fe = model.feature_engineer.transform(df)
        proba = model.predict_proba(X_fe)[0]

        top_indices = np.argsort(proba)[-top_n:][::-1]
        top_predictions = []
        for idx in top_indices:
            disease = model.label_encoder.inverse_transform([idx])[0]
            prob = float(proba[idx])
            top_predictions.append(
                {
                    "disease": disease,
                    "probability": prob,
                    "confidence": f"{prob:.2%}",
                }
            )

        best_idx = int(np.argmax(proba))
        predicted_disease = model.label_encoder.inverse_transform([best_idx])[0]
        confidence = float(proba[best_idx])

        all_probabilities = [
            (model.label_encoder.inverse_transform([i])[0], float(proba[i]))
            for i in range(len(proba))
        ]

        return {
            "animal_type": model.animal_type,
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "all_probabilities": all_probabilities,
        }

    def get_model_info(self) -> List[Dict[str, Any]]:
        models_info = []

        for animal, model_obj in [("cat", self.cat_model), ("dog", self.dog_model)]:
            info: Dict[str, Any] = {
                "animal_type": animal,
                "is_loaded": model_obj is not None,
            }
            if model_obj is not None:
                info["model_type"] = "XGBoostEnsemble"
                if (
                    hasattr(model_obj, "feature_engineer")
                    and model_obj.feature_engineer is not None
                ):
                    fe = model_obj.feature_engineer
                    if hasattr(fe, "feature_names_out") and fe.feature_names_out:
                        info["num_features"] = len(fe.feature_names_out)
            models_info.append(info)

        return models_info
