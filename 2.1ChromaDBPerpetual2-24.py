with open('example_data/paul_graham_essay.txt','r') as file:
    data = file.read()



from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer

# Maximum number of tokens in a chunk
max_tokens = 500
# Optionally can also have the splitter not trim whitespace for you
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)

chunks = splitter.chunks(data, max_tokens)

'''
#simple text splitter not using tokenizer. Just based on characters. 
from semantic_text_splitter import CharacterTextSplitter

# Maximum number of characters in a chunk
max_characters = 200
# Optionally can also have the splitter not trim whitespace for you
splitter = CharacterTextSplitter(trim_chunks=False)

chunks = splitter.chunks(data, max_characters)
'''
'''
#lines to show that chunks are being made
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: ", chunk)
'''

# List to store the embeddings, documents, metadatas, and ids
documents_list = []
metadatas_list = []
ids_list = []

for i, chunk in enumerate(chunks,start=1):
    # Assuming 'response' is a list containing the embedding for the current chunk
    documents_list.append(chunk)
    metadatas_list.append({"source": "paul_graham_essay"})
    ids_list.append(f"id{i}")
    print(f"Chunk {i} obtained successfully.")


import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="vectordb")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

#a collection is like creating a table in a DB
collection=client.get_or_create_collection(name="my_collection", embedding_function = sentence_transformer_ef)

# Use the collection.add function to add your data to Chroma. It always errors out here. Thinks I don't have an embedding model, but I don't need one...
collection.add(
    documents=documents_list,
    metadatas=metadatas_list,
    ids=ids_list
)
print("Printed all chunks to the collection!")

'''
results = collection.query(
    query_texts=["Who died and why?"],
    n_results=2,
    include=['distances', 'documents']
)

print(results)
'''


