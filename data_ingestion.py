from common.chunk_data import ChunkData
from pypdf import PdfReader
from common.env_utils import get_str_env, get_int_env
from common.genai_utils import get_vectors_as_list, dimensions, chunk_data_list

def ingest_data():
  text = load_pdf_file()
  chunks = split_text_as_chunks(text)
  vectors_list = get_vectors_as_list(chunks)
  for idx, vector in enumerate(vectors_list):
    chunk_data_list.append(ChunkData(chunks[idx], vector[0:dimensions]))

def split_text_as_chunks(text: str) -> list:
  chunk_size=get_int_env("chunk_size")
  tokens = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
  return tokens

def load_pdf_file():
  # creating a pdf reader object
  try:
    reader = PdfReader(get_str_env("input_file_path"))
    text = ''
    for idx, page in enumerate(reader.pages):
      text += page.extract_text()
    return text
  except Exception:
    reader.close()