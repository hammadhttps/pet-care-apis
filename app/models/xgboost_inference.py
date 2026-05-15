"""
Inference-only classes for XGBoost disease prediction models.
Lean versions — no training, plotting, or heavy visualization imports.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class VeterinaryFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        symptom_cols = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]

        X["symptom_count"] = X[symptom_cols].apply(
            lambda row: sum(1 for s in row if s != "No_Symptom"), axis=1
        )

        respiratory_terms = [
            "Coughing",
            "Sneezing",
            "Labored Breathing",
            "Runny Nose",
            "Nasal Discharge",
        ]
        gastrointestinal_terms = ["Vomiting", "Diarrhea", "Appetite Loss", "Nausea"]
        dermatological_terms = ["Itching", "Hair Loss", "Skin Lesions", "Redness"]
        neurological_terms = ["Seizures", "Tremors", "Disorientation", "Head Tilt"]

        def count_terms(row, terms):
            return sum(1 for col in symptom_cols if row[col] in terms)

        X["respiratory_symptom_score"] = X.apply(
            lambda row: (
                count_terms(row, respiratory_terms)
                + row["Coughing"]
                + row["Labored_Breathing"]
            ),
            axis=1,
        )
        X["gastrointestinal_symptom_score"] = X.apply(
            lambda row: (
                count_terms(row, gastrointestinal_terms)
                + row["Vomiting"]
                + row["Diarrhea"]
                + row["Appetite_Loss"]
            ),
            axis=1,
        )
        X["dermatological_flag"] = X.apply(
            lambda row: count_terms(row, dermatological_terms) > 0, axis=1
        ).astype(int)
        X["neurological_flag"] = X.apply(
            lambda row: count_terms(row, neurological_terms) > 0, axis=1
        ).astype(int)

        if "Body_Temperature_in_Celsius" in X.columns:
            temp = X["Body_Temperature_in_Celsius"]
            X["fever_severity"] = 0
            X.loc[temp > 39.2, "fever_severity"] = 1
            X.loc[temp > 40.0, "fever_severity"] = 2
            X.loc[temp > 41.0, "fever_severity"] = 3

        if "Age" in X.columns:
            X["age_group"] = pd.cut(
                X["Age"],
                bins=[0, 1, 3, 7, 15, 100],
                labels=["puppy/kitten", "young", "adult", "senior", "geriatric"],
            ).astype(str)

        X.drop(columns=symptom_cols, inplace=True)
        self.feature_names_out = X.columns.tolist()
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out


class OptimizedDiseaseTrainer:
    def __init__(self, animal_type="Dog", hierarchical=True):
        self.animal_type = animal_type
        self.hierarchical = hierarchical
        self.label_encoder = None
        self.category_encoder = None
        self.category_models = {}
        self.category_encoders = {}
        self.global_model = None
        self.flat_ensemble = None
        self.preprocessor = None
        self.feature_engineer = VeterinaryFeatureEngineer()
        self.best_params = {}
        self.feature_importances = None

    def _ensemble_predict(self, X, method="predict_proba"):
        probas = []
        for pipe in self.flat_ensemble:
            proba = (
                pipe.predict_proba(X) if method == "predict_proba" else pipe.predict(X)
            )
            probas.append(proba)
        avg = np.mean(probas, axis=0)
        if method == "predict":
            avg = np.argmax(avg, axis=1)
        return avg

    def predict_proba(self, X):
        if self.flat_ensemble:
            return self._ensemble_predict(X, "predict_proba")
        elif self.global_model:
            return self.global_model.predict_proba(X)
        else:
            raise RuntimeError("No model available.")

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
