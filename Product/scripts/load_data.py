import os
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def get_database_connection():
    """
    Establishes a connection to the PostgreSQL database.
    Includes a retry mechanism to wait for the database container to be ready.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("The DATABASE_URL environment variable is not set.")

    retries = 15
    delay_seconds = 5
    for attempt in range(retries):
        try:
            engine = create_engine(db_url)
            with engine.connect():
                print("Database connection established successfully.")
                return engine
        except OperationalError:
            print(f"Database connection failed. Retrying in {delay_seconds}s... (Attempt {attempt + 1}/{retries})")
            time.sleep(delay_seconds)

    raise ConnectionError("Could not connect to the database after multiple retries.")

def load_data_to_postgres():
    """
    Loads time-series data from a CSV file into a PostgreSQL table.
    The table is dropped and recreated on each run to ensure idempotency.
    """
    try:
        engine = get_database_connection()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, '..', 'data', 'test_data.csv')

        print(f"Attempting to read data from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['time'])
        
        table_name = 'co2_estimation_data'

        print(f"Loading {len(df)} rows into the '{table_name}' table...")
        with engine.connect() as connection:
            df.to_sql(table_name, con=connection, if_exists='replace', index=False)
            print(f"Successfully loaded data into '{table_name}'.")

            # --- NEW: VALIDATION STEP ---
            print("--- Running Validation ---")
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
            row_count = result.scalar_one()
            
            if row_count == len(df):
                print(f"VALIDATION SUCCESS: Found {row_count} rows in the database, which matches the source file.")
            else:
                print(f"VALIDATION FAILED: Found {row_count} rows in the database, but expected {len(df)}.")
            # --- END OF VALIDATION ---

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    load_data_to_postgres()