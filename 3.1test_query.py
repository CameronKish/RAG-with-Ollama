import ollama

query = "What did he work on?"
query_ollama_embedding = ollama.embeddings(model='nomic-embed-text', prompt=query)
query_ollama_embedding_data = query_ollama_embedding['embedding'] #need to take out the {'embedding',[array]} line included above.

print(query_ollama_embedding_data)