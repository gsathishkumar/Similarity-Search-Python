class ChunkData:
  def __init__(self, text : str, vector: list):
    self.text = text
    self.vector = vector

  def __str__(self):
    return f'ChunkData(\n Vector={self.vector}\n text={self.text}\n)'