from pypdf import PdfReader
from env_utils import EnvironmentUtils
from genai_utils import GenAiClient
from chunk_data import ChunkMetaData

class DataIngestion:

  def __init__(self, genai_client : GenAiClient):
    self._genai_client = genai_client

  def ingest_data(self):
    text = self.load_pdf_file()
    chunks = self.split_text_as_chunks(text)
    vectors_list = self._genai_client.get_vectors_as_list(chunks)
    for idx, vector in enumerate(vectors_list):
      self._genai_client.add_chunk(ChunkMetaData(chunks[idx], vector[0:self._genai_client.dimensions]))

  def split_text_as_chunks(self, text: str) -> list:
    chunk_size=EnvironmentUtils.get_int_env("chunk_size")
    tokens = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return tokens

  def load_pdf_file(self):
    # creating a pdf reader object
    try:
      reader = PdfReader(EnvironmentUtils.get_str_env("input_file_path"))
      text = ''
      for idx, page in enumerate(reader.pages):
        text += page.extract_text()
      return text
    except Exception:
      reader.close()