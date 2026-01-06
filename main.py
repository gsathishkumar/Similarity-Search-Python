from dotenv import load_dotenv
from data_ingestion import DataIngestion
from query_processing import QueryProcess
from genai_utils import GenAiClient

# Load variables from the .env file
load_dotenv()

if __name__ == '__main__':
  genai_client = GenAiClient(1024, [])
  DataIngestion(genai_client).ingest_data()
  input_text = input('Enter Search Text: ')
  QueryProcess(genai_client).process_text(input_text)