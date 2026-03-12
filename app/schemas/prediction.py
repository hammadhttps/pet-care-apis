from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum

class AnimalType(str, Enum):
    CAT = "Cat"
    DOG = "Dog"

class Sex(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class PetHealthData(BaseModel):
    Animal_Type: str = Field(..., description="Type of animal (Cat or Dog)")
    Sex: str = Field(..., description="Sex of the animal")
    Breed: str = Field(..., description="Breed of the animal")
    Age: float = Field(..., gt=0, lt=30, description="Age in years")
    Weight: float = Field(..., gt=0, lt=100, description="Weight in kg")
    Symptom_1: str = Field(..., description="Primary symptom")
    Symptom_2: str = Field(..., description="Secondary symptom")
    Symptom_3: str = Field(..., description="Tertiary symptom")
    Symptom_4: str = Field(..., description="Quaternary symptom")
    Appetite_Loss: int = Field(0, ge=0, le=1, description="Appetite loss (0 or 1)")
    Vomiting: int = Field(0, ge=0, le=1, description="Vomiting (0 or 1)")
    Diarrhea: int = Field(0, ge=0, le=1, description="Diarrhea (0 or 1)")
    Coughing: int = Field(0, ge=0, le=1, description="Coughing (0 or 1)")
    Labored_Breathing: int = Field(0, ge=0, le=1, description="Labored breathing (0 or 1)")
    Body_Temperature_in_Celsius: float = Field(..., gt=35, lt=43, description="Body temperature in Celsius")

    @validator('Animal_Type')
    def validate_animal_type(cls, v):
        if v.lower() not in ['cat', 'dog']:
            raise ValueError('Animal type must be either Cat or Dog')
        return v.capitalize()

    @validator('Sex')
    def validate_sex(cls, v):
        if v.lower() not in ['male', 'female']:
            raise ValueError('Sex must be either Male or Female')
        return v.capitalize()

    @validator('Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', 'Labored_Breathing')
    def validate_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError('Value must be 0 or 1')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Animal_Type": "Cat",
                "Sex": "Female",
                "Breed": "Himalayan",
                "Age": 1.9,
                "Weight": 5.6,
                "Symptom_1": "Vaginal Discharge",
                "Symptom_2": "Abdominal Pain",
                "Symptom_3": "Lethargy",
                "Symptom_4": "Fever",
                "Appetite_Loss": 1,
                "Vomiting": 0,
                "Diarrhea": 0,
                "Coughing": 0,
                "Labored_Breathing": 0,
                "Body_Temperature_in_Celsius": 38.5
            }
        }

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: Optional[float] = None
    message: str
    data: Optional[dict] = None

class HealthCheck(BaseModel):
    status: str
    message: str

class ModelInfo(BaseModel):
    animal_type: str
    is_loaded: bool
    num_features: Optional[int] = None
    model_type: Optional[str] = None
