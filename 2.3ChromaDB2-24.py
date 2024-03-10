#Purpose: To create a Chroma database to store our file embeddings

#Simple load method for one file.
with open('example_data/paul_graham_essay.txt','r') as file:
    data = file.read()


##2 Text Splitting options below. Tokenizer is more precise. 
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
max_characters = 100
# Optionally can also have the splitter not trim whitespace for you
splitter = CharacterTextSplitter(trim_chunks=False)

chunks = splitter.chunks(data, max_characters)
'''
'''
#lines to print that chunks are being made
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: ", chunk)
'''

# List to store the embeddings, documents, metadatas, and ids
documents_list = []
metadatas_list = []
ids_list = []

#Only going to make the first 5 chunks to save time
for i, chunk in enumerate(chunks[:5],start=1):
    # Assuming 'response' is a list containing the embedding for the current chunk
    documents_list.append(chunk)
    metadatas_list.append({"source": "paul_graham_essay"})
    ids_list.append(f"id{i}")
    print(f"Chunk {i} obtained successfully.")


import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.Client()

#Calling Chromadb to use a model I choose to embed both the files above and the query below. 
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

#a collection is like creating a table in a DB
collection=chroma_client.create_collection(name="my_collection", embedding_function = sentence_transformer_ef)

# Use the collection.add function to add your data to Chroma. It will error out here if you don't have Python 10.10 in use.
collection.add(
    documents=documents_list,
    metadatas=metadatas_list,
    ids=ids_list
)
print("Printed all chunks to the collection!")

#Will use the model from the sentence transformer above to embed the question and then compare against the database. 
results = collection.query(
    query_texts=["What did he go to college for?"],
    n_results=5,
    include=['documents']
)

print(results)
