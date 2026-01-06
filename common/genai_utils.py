from common.env_utils import get_str_env
from google import genai

dimensions = 1024
chunk_data_list = []

def get_vectors_as_list(chunks = []) -> list :
  try:
    client = genai.Client(api_key=get_str_env("GEMINI_API_KEY"))
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=chunks
    )
    vectors_list = [embedding_content.values for embedding_content in response.embeddings]
    return vectors_list # Returns list of Vectors Array
  except Exception as e:
    print('-'*40, '\n' , e)
    client.close()