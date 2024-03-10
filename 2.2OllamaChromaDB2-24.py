#Purpose: To create a ChromaDB with Ollama embeddings

with open('example_data/paul_graham_essay.txt','r') as file:
    data = file.read()


from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer

##Tokenizer text splitter
# Maximum number of tokens in a chunk
max_tokens = 1000
# Optionally can also have the splitter not trim whitespace for you
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)

chunks = splitter.chunks(data, max_tokens)
##

'''
##Used for a simple text splitter
#simple text splitter not using tokenizer. Just based on characters. 
from semantic_text_splitter import CharacterTextSplitter

# Maximum number of characters in a chunk
max_characters = 200
# Optionally can also have the splitter not trim whitespace for you
splitter = CharacterTextSplitter(trim_chunks=False)

chunks = splitter.chunks(data, max_characters)
##
'''
'''
#lines to show that chunks are being made
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: ", chunk)
'''
#For use to make fun percentage completion of embedding.
total_chunks=len(chunks)

import ollama


# List to store the embeddings, documents, metadatas, and ids
embeddings_list = []
documents_list = []
metadatas_list = []
ids_list = []


# Loop through your chunks and generate embeddings
for i, chunk in enumerate(chunks, start=1):
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
        #This response includes a very annoying {'embedding',[array]} line. We only want the array. The below only takes the array before passing it to the list.
        embedding_data = response['embedding']

        # Add data to the lists
        embeddings_list.append(embedding_data)
        documents_list.append(chunk)
        metadatas_list.append({"source": "my_source"})
        ids_list.append(f"id{i}")
        print(f"Embedding for chunk {i} obtained successfully. {(i/total_chunks)*100:.1f} percent complete.")
    except Exception as e:
        print(f"Error processing chunk {i}: {str(e)}")


import chromadb
chroma_client = chromadb.Client()

#a collection is like creating a table in a DB. An embedding_function can be added here. This will be used for querying (and embedding I assume)?
collection=chroma_client.create_collection(name="my_collection")    #,metadata={"hnsw:space": "cosine"}) # l2 is the default distance function. This code will swap it for cosine.

# Use the collection.add function to add your data to Chroma. Make sure you are using python version 10.10 otherwise this will error out. 
collection.add(
    embeddings=embeddings_list,
    documents=documents_list,
    metadatas=metadatas_list,
    ids=ids_list
)

query = "Who died and why?"
query_ollama_embedding = ollama.embeddings(model='nomic-embed-text', prompt=query)
query_ollama_embedding_data = query_ollama_embedding['embedding'] #need to take out the {'embedding',[array]} line included above.


results = collection.query(
    query_embeddings=query_ollama_embedding_data,
    n_results=3,
    include=['documents','distances']
)

print(results)

#Split results into only the documents we need to feed to Ollama
docs_only = results['documents']

#print(docs_only)

#Create content for Ollama (whichever model I choose) to answer
content= "You are being provided context to help answer a question. Not all context may be relevant: " + str(docs_only) + "\n\n\n Using that information as context, please answer the following question in 100 words or less: " + query

#To connect Ollama to the data and come up with an answer.
stream = ollama.chat(
    model='llama2:7b-chat-q4_K_M',
    messages=[{'role': 'user', 'content': content}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
