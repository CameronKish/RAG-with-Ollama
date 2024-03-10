#Purpose: Now that you have a database created, this file lets you query it without trying to embed everything again. 

import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="vectordb")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

#a collection is like creating a table in a DB
collection=client.get_collection(name="my_collection", embedding_function = sentence_transformer_ef)

query = "In what context are aliens talked about in this document?"

results = collection.query(
    query_texts=[query],
    n_results=1,
    include=['distances', 'documents']
)

print(results)

#Split results into only the documents we need to feed to Ollama
docs_only = results['documents']

#print(docs_only)

#Create content for Ollama (whichever model I choose) to answer. Can use specialized prompts if using smaller models.
content= "You are a helpful AI assitant specializing in reviewing informaion and providing thoughtful and concise answers. \n\n You will be given a chunk of contextual information to help you answer a question which is provided after. Do your best with the context you have to answer the question. Not all context may be relevant: " + str(docs_only) + "\n\n\n Using that information as context, please answer a query in 100 words or less. Keep your sentences brief. Do not acknowledge the request. Do not deviate from the given request. Here is the question: " + query


import ollama

#To connect Mistral to the data and come up with an answer.
stream = ollama.chat(
    model='mistral:7b-instruct',
    messages=[{'role': 'user', 'content': content}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
