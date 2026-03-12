# Pets Disease Prediction API

FastAPI service that predicts cat/dog diseases from basic pet info and symptoms.

## Run locally

```bash
python run.py
```

Server runs at `http://127.0.0.1:8000`.

## Endpoints

- `GET /`: health check
- `GET /models`: list model load status
- `POST /predict/cat`: cat disease prediction
- `POST /predict/dog`: dog disease prediction
- `POST /predict`: auto-detect by `Animal_Type`

## Example request (cat)

### PowerShell (recommended)

```powershell
@'
{
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
'@ | Out-File -Encoding utf8 body.json

curl.exe -X POST "http://127.0.0.1:8000/predict/cat" `
  -H "Content-Type: application/json" `
  --data-binary "@body.json"
```

