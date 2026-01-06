from env_utils import EnvironmentUtils
from chunk_data import ChunkMetaData
from google import genai

class GenAiClient:
    def __init__(self, dimensions: int, chunk_meta_data: list[ChunkMetaData]):
        self._dimensions = dimensions
        self._chunk_data = list(chunk_meta_data) if chunk_meta_data is not None else []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: int):
        if not isinstance(value, int):
            raise TypeError("dimensions must be an int")
        self._dimensions = value

    @property
    def chunk_data(self) -> list[ChunkMetaData]:
        # return a copy to prevent direct modification
        return list(self._chunk_data)

    def add_chunk(self, chunk_meta_data: ChunkMetaData) -> None:
        self._chunk_data.append(chunk_meta_data)

    def get_vectors_as_list(self, chunks = []) -> list :
      try:
        client = genai.Client(api_key=EnvironmentUtils.get_str_env("GEMINI_API_KEY"))
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=chunks
        )
        vectors_list = [embedding_content.values for embedding_content in response.embeddings]
        return vectors_list # Returns list of Vectors Array
      except Exception as e:
        print('-'*40, '\n' , e)
        client.close()