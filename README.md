# Similarity-Search-Python

Similarity search in Python involves reading pdf, convert text into numerical vectors (embeddings) and then find the closest vectors to a given query based on similarity search using numpy library.

## General Workflow

### Data Ingestion

- Load Data: Read the PDF and get the text contents
- Generate Embeddings: Convert the text chunks into numerical vectors using an embedding model (e.g., Google GEN-AI).

### Query Processing

- Perform Search: When a user provides a query, convert the query into an embedding, and search the vector store to find the nearest neighbors (most similar results).
