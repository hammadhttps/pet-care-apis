from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from app.schemas.prediction import (
    PetHealthData,
    PredictionDetailResponse,
    HealthCheck,
    ModelInfo,
)
from app.services.prediction_service import PredictionService
from app.utils.validators import validate_pet_data

# Initialize FastAPI app
app = FastAPI(
    title="Pets Disease Prediction API",
    description="API for predicting diseases in cats and dogs based on symptoms",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
MODELS_PATH = "models/"
prediction_service = PredictionService(MODELS_PATH)


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        prediction_service.load_models()
        print("Models loaded successfully")
        if prediction_service.cat_model:
            print(
                f"  Cat model: {type(prediction_service.cat_model).__name__} ({type(prediction_service.cat_model).__module__})"
            )
        if prediction_service.dog_model:
            print(
                f"  Dog model: {type(prediction_service.dog_model).__name__} ({type(prediction_service.dog_model).__module__})"
            )
    except Exception as e:
        print(f"Error loading models: {e}")


@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy", message="Pets Disease Prediction API is running"
    )


@app.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get information about available models"""
    return prediction_service.get_model_info()


@app.post("/predict/cat", response_model=PredictionDetailResponse)
async def predict_cat_disease(
    pet_data: PetHealthData, _validated_data: dict = Depends(validate_pet_data)
):
    """
    Predict disease for cats based on provided symptoms and health data
    """
    try:
        if not prediction_service.cat_model:
            raise HTTPException(
                status_code=503,
                detail="Cat model not available. Please try again later.",
            )

        result = prediction_service.predict(
            animal_type="cat", data=pet_data.model_dump()
        )

        return PredictionDetailResponse(
            success=True,
            animal_type=result["animal_type"],
            predicted_disease=result["predicted_disease"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            all_probabilities=result["all_probabilities"],
            message="Prediction completed successfully",
            data=pet_data.model_dump(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict/dog", response_model=PredictionDetailResponse)
async def predict_dog_disease(
    pet_data: PetHealthData, _validated_data: dict = Depends(validate_pet_data)
):
    """
    Predict disease for dogs based on provided symptoms and health data
    """
    try:
        if not prediction_service.dog_model:
            raise HTTPException(
                status_code=503,
                detail="Dog model not available. Please try again later.",
            )

        result = prediction_service.predict(
            animal_type="dog", data=pet_data.model_dump()
        )

        return PredictionDetailResponse(
            success=True,
            animal_type=result["animal_type"],
            predicted_disease=result["predicted_disease"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            all_probabilities=result["all_probabilities"],
            message="Prediction completed successfully",
            data=pet_data.model_dump(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict", response_model=PredictionDetailResponse)
async def predict_disease(
    pet_data: PetHealthData, _validated_data: dict = Depends(validate_pet_data)
):
    """
    Auto-detect animal type and predict disease
    """
    try:
        animal_type = pet_data.Animal_Type.lower()

        if animal_type not in ["cat", "dog"]:
            raise HTTPException(
                status_code=400, detail="Animal type must be either 'cat' or 'dog'"
            )

        model = (
            prediction_service.cat_model
            if animal_type == "cat"
            else prediction_service.dog_model
        )
        if not model:
            raise HTTPException(
                status_code=503,
                detail=f"{animal_type.capitalize()} model not available",
            )

        result = prediction_service.predict(
            animal_type=animal_type, data=pet_data.model_dump()
        )

        return PredictionDetailResponse(
            success=True,
            animal_type=result["animal_type"],
            predicted_disease=result["predicted_disease"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            all_probabilities=result["all_probabilities"],
            message="Prediction completed successfully",
            data=pet_data.model_dump(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )
