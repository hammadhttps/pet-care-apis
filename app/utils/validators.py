from fastapi import HTTPException
from app.schemas.prediction import PetHealthData
from typing import Dict, Any


async def validate_pet_data(pet_data: PetHealthData) -> Dict[str, Any]:
    """
    Additional validation for pet health data.
    """
    data_dict = pet_data.dict()

    animal_type = data_dict["Animal_Type"].lower()
    age = data_dict["Age"]

    if animal_type == "cat" and age > 25:
        raise HTTPException(
            status_code=400,
            detail="Invalid age: Cats typically live up to 25 years",
        )
    if animal_type == "dog" and age > 20:
        raise HTTPException(
            status_code=400,
            detail="Invalid age: Dogs typically live up to 20 years",
        )

    weight = data_dict["Weight"]
    if animal_type == "cat" and (weight < 1 or weight > 15):
        raise HTTPException(
            status_code=400,
            detail="Invalid weight for cat: Should be between 1-15 kg",
        )
    if animal_type == "dog" and (weight < 1 or weight > 90):
        raise HTTPException(
            status_code=400,
            detail="Invalid weight for dog: Should be between 1-90 kg",
        )

    temp = data_dict["Body_Temperature_in_Celsius"]
    if temp < 37 or temp > 39.5:
        raise HTTPException(
            status_code=400,
            detail=f"Abnormal body temperature: {temp} C. Normal range is 37-39.5 C",
        )

    return data_dict
