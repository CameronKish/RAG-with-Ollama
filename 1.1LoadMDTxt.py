#Purpose: to be able to load markdown and txt files. Depending on file token length, embeddings are either 1 unit or separated.

max_tokens = 500 #global token length we want

#A langchain alternative to load everything. Could really just do the regular load method. 
from langchain_community.document_loaders import TextLoader

loader = TextLoader("example_data/DWYL6.md")
data = loader.load()

# Extract the text content from the Document object
page_content = data[0].page_content

from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer

def process_page(page_content):
    ##Tokenizer text splitter
    # # Optionally can also have the splitter not trim whitespace for you
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)
    chunks = splitter.chunks(page_content, max_tokens)
    ##
    # Check the length of page_content
    if len(chunks) > 1:
        # Do something if page_content is greater than 500 characters
        print("Processing long page content:...")
        return chunks

    else:
        # Return or do something else if page_content is not greater than 500 characters
        print("Keeping short page content.")
        return page_content


print(process_page(page_content))
# Print the text content
#print(len(page_content))
