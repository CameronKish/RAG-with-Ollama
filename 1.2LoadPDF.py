from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
import ollama

loader = PyPDFLoader("example_data/ChatGPT_FPA.pdf")
pages = loader.load_and_split()


# Lists to store page content, metadata, and document IDs
pdf_embedding_list = []
pdf_page_content_list = []
pdf_metadata_list = []
pdf_id_list = []

# Iterate through each page
for page in pages:
    # Extract page content and metadata. The structure of pages[0] is an object with page_content and metadata fields.
    pdf_page_content = page.page_content
    pdf_metadata = page.metadata
    # Generate a timestamp-based document ID
    pdf_id = str(int(datetime.now().timestamp()))

    response = ollama.embeddings(model='nomic-embed-text', prompt=pdf_page_content)
    #This response includes a very annoying {'embedding',[array]} line. We only want the array. The below only takes the array before passing it to the list.
    pdf_embedding = response['embedding']
    # Add data to the embedding list
    pdf_embedding_list.append(pdf_embedding)

    # Append to respective lists
    pdf_page_content_list.append(pdf_page_content)
    pdf_metadata_list.append(pdf_metadata)
    pdf_id_list.append(pdf_id)



    # Print the result for demonstration purposes
    print(f"Embedding: {pdf_embedding}")
    #print(f"Document ID: {pdf_id}")
    #print(f"Page Content: {page_content}")
    #print(f"Metadata: {pdf_metadata}\n")

