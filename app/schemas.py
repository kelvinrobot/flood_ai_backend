from pydantic import BaseModel
from typing import List

class FloodRequest(BaseModel):
    Average_temp: float
    humidity: float
    precip: float
    windspeed: float
    sealevelpressure: float
    cloudcover: float
    solarradiation: float
    severerisk:float
    flood_lag_1: int
    flood_lag_2: int
    flood_lag_3: int
    flood_lag_4: int
    flood_lag_5: int
    SMI_linear_norm: float
    month: int
    

class FloodResponse(BaseModel):
    flood_probability_percent: float  # e.g. 81.25
    flood_risk_score_percent: float   # e.g. 58.33
    severity_class: str               # "low", "mid", "high"
    model_votes: List[str]
    final_flood: bool
