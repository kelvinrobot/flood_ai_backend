

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
from app.model_fusion import predict_flood
from app.schemas import FloodRequest, FloodResponse
from app.retrain import retrain_all_models

app = FastAPI()

@app.post("/predict", response_model=FloodResponse)
def predict_endpoint(payload: FloodRequest):
    return predict_flood(payload)

@app.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "File must be a CSV."})

    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
        retrain_all_models(df)
        return {"message": " Models retrained successfully with uploaded dataset."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
