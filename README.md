# Time Series Transformer and API Deployment

A production-ready pipeline for CO₂ concentration forecasting using a Transformer model, deployed with FastAPI, PostgreSQL, and Docker.

## Table of Contents

- [Features](#features)
- [Quick Start: Deployment with Docker](#quick-start-deployment-with-docker)
- [API Usage](#api-usage)
- [Interacting with the Database](#interacting-with-the-database)
- [Local Development & Training](#local-development--training)
- [Model & Data](#model--data)


## Features

- **Transformer-Based Model**: Utilizes Transformer architecture for time-series forecasting.
- **API**: A robust API built with FastAPI to serve predictions.
- **PostgreSQL Integration**: Loads and serves data from a PostgreSQL database.
- **Dockerized Pipeline**: Fully containerized setup using Docker Compose for one-command deployment.
- **Data Analysis**: Includes notebooks for data processing and in-depth model performance analysis.

## Quick Start: Deployment with Docker

This is the recommended method for running the project. It handles the database, data loading, and API server automatically.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/XC9292/Time-Series-Transformer-and-API-Deployment.git
   cd Time-Series-Transformer-and-API-Deployment/Product
   ```

2. **Build and Run with Docker Compose:** This single command will build the images, start the containers, wait for the database to be healthy, load the data, and start the API server.

   ```bash
   docker compose up --build
   ```

   The API will be available at `http://localhost:8000`.

3. **Managing the Application:**

   - To run the containers without rebuilding in the background (detached mode) or foreground: `docker compose up -d` or `docker compose up`.
   - To stop the application: `docker compose down` (or `Ctrl+C` if running in the foreground).

## API Usage

The API provides an endpoint to get CO₂ estimations based on a given timestamp. It selects all measurements up to and including the provided timestamp, preprocesses the last 15 data points, and feeds them to the model.

You can test the endpoint using `curl` or refer to the `api_test.ipynb` notebook for a Python example.

**Example Request:**

```bash
curl -X GET "http://localhost:8000/predict/timestamp?timestamp=2014-02-07T12:43:50"
```

<details> <summary><strong>✅ Expected Response</strong></summary>

```
{
 "predicted_value":{
   "CO2@1 (%)":0.038093964644117864,
   "CO2@2 (%)":0.033151381406842734,
   "CO2@3 (%)":0.06567349688226769,
   "CO2@4 (%)":0.08197609072274116,
   "CO2@5 (%)":0.566436955544155,
   "CO2@6 (%)":3.1849114810699124
   },
  "source_timestamp": [
   "2014-02-07T12:33:36",
   "2014-02-07T12:34:19",
   "2014-02-07T12:35:02",
   "2014-02-07T12:35:46",
   "2014-02-07T12:36:29",
   "2014-02-07T12:37:12",
   "2014-02-07T12:37:55",
   "2014-02-07T12:38:38",
   "2014-02-07T12:39:22",
   "2014-02-07T12:40:05",
   "2014-02-07T12:40:48",
   "2014-02-07T12:41:31",
   "2014-02-07T12:42:14",
   "2014-02-07T12:42:58",
   "2014-02-07T12:43:41"
   ],
  "data_points_used": 15
}
```

</details>

## Interacting with the Database

Once the Docker containers are running, you can connect directly to the PostgreSQL database to view and analyze the test data.

1. **Connect to the Database Container:** Open a new terminal and run:

   ```bash
   docker exec -it timeseries_db psql -U user -d mydatabase
   ```

2. **List Tables:** Run `\dt` to see the loaded tables.

   <details> <summary><strong>✅ Expected Output</strong></summary>

   ```bash
                List of relations
    Schema |       Name        | Type  | Owner
   --------+---------------------+-------+-------
    public | co2_estimation_data | table | user
   (1 row)
   ```

   </details>

3. **Run SQL Queries:** You can now query the data directly.

   ```SQL
   SELECT time, "6_sampling" FROM co2_estimation_data
   WHERE time > '2014-02-07 12:00:50'
   ORDER BY time DESC
   LIMIT 5;
   ```

   <details> <summary><strong>✅ Expected Output</strong></summary>

   ```bash
           time         |     6_sampling
   ---------------------+--------------------
    2014-02-07 13:20:24 | 0.2653370295911377
    2014-02-07 13:19:41 | 0.2653370295911377
    2014-02-07 13:18:58 | 0.2986724783768284
    2014-02-07 13:18:14 | 0.2971634953104303
    2014-02-07 13:17:31 | 0.2956545122440322
   (5 rows)
   ```

   </details>

## Local Development & Training

Follow these steps if you want to modify the model or run the training process locally.

1. **Clone the repository and navigate to the project directory.**

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model:** Run the training script with your desired configuration (from the `cfg/` directory) and window size.

   ```bash
   python train_transformer.py --cfg transformer_selected_feature --ws 15
   ```

## Model & Data

- **Model Architecture**: The Transformer model is defined in `model.py`, and the training loop is managed in `trainer.py`.
- **Data Processing**: The raw data in the `data/` directory is processed using the `data_process.ipynb` notebook.
- **Model Analysis**: A deep dive into the model's performance, including visualizations and **ROOT CASUE ANALYSIS**, is available in `model_analysis.ipynb`.
