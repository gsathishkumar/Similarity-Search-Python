import numpy as np
from common.chunk_data import ChunkData
from common.genai_utils import get_vectors_as_list, dimensions, chunk_data_list

def process_query():
  text, vector = get_vectors_for_user_input()
  query_chunk_data = ChunkData(text, vector[0:dimensions])
  similarity_list = get_similarity_for_each_chunk(query_chunk_data, chunk_data_list)
  similarity_list.sort(key=lambda x: x[0], reverse=True) # Sort By Similarity
  for idx, item_tuple  in enumerate(similarity_list[:5], start=1):
    print(f"Top: {idx}, Text: {item_tuple [1].text}")
  
def get_similarity_for_each_chunk(query_chunk_data: ChunkData, chunk_data_list: list[ChunkData]) -> list[tuple]:
  similarity_all = []
  for item_chunk_data in chunk_data_list:
    similarity = cosine_similarity_numpy(query_chunk_data.vector , item_chunk_data.vector)
    similarity_all.append((similarity, item_chunk_data))
  return similarity_all

def cosine_similarity_numpy(vec1, vec2):
    # Ensure inputs are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Check for zero vectors (magnitude 0) to avoid division by zero
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0 # Or raise an error, depending on needs

    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    return similarity

def get_vectors_for_user_input():
  input_text = input('Enter Search Text: ')
  vectors_list = get_vectors_as_list([input_text])
  # vectors_list[0] --> Multi-dimensional Vector for given text
  return (input_text, vectors_list[0])