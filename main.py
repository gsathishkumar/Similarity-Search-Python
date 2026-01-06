from dotenv import load_dotenv
from data_ingestion import ingest_data
from query_processing import process_query

# Load variables from the .env file
load_dotenv()

if __name__ == '__main__':
  ingest_data()
  process_query()