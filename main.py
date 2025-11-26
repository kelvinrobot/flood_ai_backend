from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from app.model_fusion import predict_flood
from app.schemas import FloodRequest, FloodResponse
from app.retrain import retrain_all_models

app = FastAPI()

# Comprehensive CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explicitly list all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"],  # Expose all headers to browser
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Add global OPTIONS handler for all endpoints
@app.options("/{rest_of_path:path}")
async def options_handler(rest_of_path: str):
    return JSONResponse(
        content={"message": "CORS preflight"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "600"
        }
    )

@app.get("/")
def read_root():
    return {"message": "Flood AI backend is live and running!"}

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

# Add response headers to all responses
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response
