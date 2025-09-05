# Time Series Transformer and API Deployment

Implementation of a transformer-based time series model for CO2 concentration estimation and its production-ready deployment.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Deployment](#deployment)
    - [REST Endpoint](#rest-endpoint)
    - [PostgreSQL](#postgresql)
- [Model](#model)
- [Data](#data)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/XC9292/Time-Series-Transformer-and-API-Deployment.git
   cd Time-Series-Transformer-and-API-Deployment
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, you can run the `train_transformer.py` script with the desired configuration file (I have provided two configuration files in `cfg/`).

```bash
python train_transformer.py --cfg <config_name> --ws <window_size>
```

For example:
```bash
python train_transformer.py --cfg transformer --ws 10
```

### Deployment

The model can be deployed using Docker. A `Dockerfile` and `docker-compose.yaml` are provided in the `Product/` directory for easy deployment.

To build and run the Docker container:
```bash
cd Product
```

Dockerize the pipeline (Postgres + FastAPI + preprocessing) via:
```bash
docker compose up --build
```

#### REST Endpoint
For API prediction, you just need to enter the time stamp in test data, such as `2014-02-07 12:04:05`. Then, the API can select all measurements before and include this data via:
```SQL
SELECT * FROM co2_estimation_data
WHERE time <= :ts
ORDER BY time;
```
The pre-processing function is implemented in `Product/app/main.py` to select the measurements based on the pre-defined window size (15 in API) and convert to `features` for model's prediction.

The API will be available at `http://0.0.0.0:8000`. You can refer to `api_test.ipynb` to test our dockerized API.
```Python
import requests


url = "http://localhost:8000/predict/timestamp"
params = {
    "timestamp": "2014-02-07 12:43:50"
}

%time response = requests.get(url=url, params=params)

output = response.json()
for key, value in output.items():
    print(f"{key}: {value}")
```
The output is:
```txt
CPU times: user 4.86 ms, sys: 0 ns, total: 4.86 ms
Wall time: 38.3 ms
predicted_value: {'CO2@1 (%)': 0.003164813155308366, 'CO2@2 (%)': 0.0027541876770555973, 'CO2@3 (%)': 0.005456096492707729, 'CO2@4 (%)': 0.006810501683503389, 'CO2@5 (%)': 0.047059085220098495, 'CO2@6 (%)': 0.26459965109825134}
source_timestamp: ['2014-02-07T12:33:36', '2014-02-07T12:34:19', '2014-02-07T12:35:02', '2014-02-07T12:35:46', '2014-02-07T12:36:29', '2014-02-07T12:37:12', '2014-02-07T12:37:55', '2014-02-07T12:38:38', '2014-02-07T12:39:22', '2014-02-07T12:40:05', '2014-02-07T12:40:48', '2014-02-07T12:41:31', '2014-02-07T12:42:14', '2014-02-07T12:42:58', '2014-02-07T12:43:41']
data_points_used: 15
```
#### PostgreSQL

The processed test data `processed_data/test_data.csv` is loaded into PostgreSQL after running above docker set up. You can open another terminal (**keep docker on**) and easily access this schema via:
```bash
docker exec -it timeseries_db psql -U user -d mydatabase
```
Then you can run:
```bash
\dt
```
The output should be:
```bash
              List of relations
 Schema |        Name         | Type  | Owner 
--------+---------------------+-------+-------
 public | co2_estimation_data | table | user
(1 row)
```
Then you can run SQL in the command to access and analyze the test data.



## Model

This project uses a Transformer-based model for time series forecasting. The model architecture is defined in `model.py`. The training process is handled by the `trainer.py` script, which includes the training loop, evaluation, and checkpointing.

### Analysis of Model
The performance analysis of the trained model is in `model_analysis.ipynb`, including visualization and **ROOT CAUSE ANALYSIS**.

## Data

The dataset used for this project is located in the `data/` directory. The data is preprocessed using the `data_process.ipynb` notebook. The processed data is stored in the `processed_data/` directory.
