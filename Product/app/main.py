import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
from sqlalchemy import create_engine, text
from scripts.model import TimeSeriesTransformer
import torch
import numpy as np

# --- Database Connection ---
# Default to a standard local setup if the env var is not set.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/mydatabase")
engine = create_engine(DATABASE_URL)
CO2_scale = 12.03671837

# --- Setup Model ---
window_sizes = 15
model_path = "./model/model.pth.tar"
model = TimeSeriesTransformer(
        input_dim=38,
        d_model=192,
        n_heads=3,
        n_layers=3,
        d_ff=384,
        max_seq_len=window_sizes,
        output_dim=6,
        dropout=0.1
    )

ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Time Series Prediction API",
    description="Provides predictions from a model based on time-series data stored in PostgreSQL."
)

# --- Pydantic Models for API Data Structure ---
class PredictionResponse(BaseModel):
    predicted_value: dict = Field(..., example={"CO2@1 (%)": 0.1}, description="CO2 predictions at 6 sampling points.")
    source_timestamp: list = Field(..., example=["2014-02-07 11:56:10"], description="Timestamps of the data point used for prediction.")
    data_points_used: int = Field(1, example=1, description="Number of data points used to make the prediction.")

# --- Data Preprocessing Function ---
def df_preprocess(df: pd.DataFrame) -> torch.Tensor:
    """
    Preprocess the dataframe to extract features and convert them to a torch tensor.
    """

    # Define feature columns
    feature_columns = ['TT302(0C)','TT300(0C)','TT401(0C)','TT303(0C)','TT400(0C)','TT113(0C)',
                        'TT301(0C)','TT202(0C)','TT412(0C)','TT309(0C)','TT110a(0C)','TT410(0C)',
                        'TT112(0C)','PT103(barg)','PT402(barg)','PT401(barg)','TT404(0C)', 'TT214(0C)',
                        'FT304(kg/hr)','AT100(pH)','FT303m3/hr','AT300(pH)','FT105(L/min)','FT103(kg/hr)',
                        'PT403(barg)','TT304(0C)','FT301m3/hr','TT107(0C)', 'PT110(barg)','PT111(barg)',
                        'TT210(0C)','TT211(0C)']
    # extract features and labels
    features = df[feature_columns].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)

    # convert labels (1-6) to one-hot encoding (6 classes)
    labels_onehot = np.eye(6)[labels - 1].astype(np.float32)  # Subtract 1 to make it 0-5 indexed
    # generate the final feature set
    features = np.concatenate([features, labels_onehot], axis=-1)
    
    # convert to torch tensor
    features = torch.from_numpy(features)
        
    return features

# --- "Trained Model" Logic ---
def get_prediction(df: pd.DataFrame) -> dict:
    """
    Get prediction from the model based on the input dataframe.
    Args:
        df: DataFrame containing the input data for prediction.
    Returns:
        results: Dictionary containing the prediction results.
        num_callback: Number of data points used for the prediction.
    """

    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for prediction.")
        
    features = df_preprocess(df)
    num_data = features.shape[0]
    if num_data < window_sizes:
        features = features.unsqueeze(0)
    else:
        features = features[-window_sizes:].unsqueeze(0)
    with torch.no_grad():
        prediction = model(features, use_causal_mask=True)
        prediction = prediction.cpu().numpy()[0]
        num_callback = window_sizes - 1
        
        results = {
            "CO2@1 (%)": prediction[0].item()*CO2_scale,
            "CO2@2 (%)": prediction[1].item()*CO2_scale,
            "CO2@3 (%)": prediction[2].item()*CO2_scale,
            "CO2@4 (%)": prediction[3].item()*CO2_scale,
            "CO2@5 (%)": prediction[4].item()*CO2_scale,
            "CO2@6 (%)": prediction[5].item()*CO2_scale,
        }

    return results, num_callback

# --- API Endpoints ---
@app.get("/", summary="Root Endpoint")
def read_root():
    """A simple welcome message to verify the API is running."""
    return {"message": "Welcome to the Prediction API."}

@app.get("/predict/timestamp", response_model=PredictionResponse, summary="Predict for a Single Timestamp")
def predict_at_timestamp(timestamp: datetime = Query(..., description="Prediction at specific time step, such as 2014-02-07 11:56:10.")):
    """
    Predict CO2 values at 6 sampling points for a given timestamp.
    The API fetches the data point earlier than the provided time stamp from the database and uses it for prediction.
    Args:
        timestamp: The specific time step for which to make the prediction.
    Returns:
        A dictionary containing the predicted CO2 values, source timestamp, and number of data points used.
    """
    try:
        # This SQL query finds the row with the minimum time difference
        # from the specified timestamp. It's an efficient way to find the nearest point.
        query = text("""
            SELECT * FROM co2_estimation_data
            WHERE time <= :ts
            ORDER BY time;
        """)
        with engine.connect() as connection:
            result_df = pd.read_sql(query, connection, params={'ts': timestamp})

        if result_df.empty:
            raise HTTPException(status_code=404, detail="No data found in the database near the specified timestamp.")

        prediction, num_callback = get_prediction(result_df)

        

        return JSONResponse({
                            "predicted_value": prediction,
                            "source_timestamp": result_df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S').to_list() if len(result_df) < num_callback+1 else result_df['time'].iloc[ - (num_callback + 1):].dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
                            "data_points_used": num_callback + 1
                            })
    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")