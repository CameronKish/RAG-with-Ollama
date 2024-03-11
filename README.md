# RAG-with-Ollama
Retrieval Augmented Generation (RAG) is what gives small LLMs with small context windows the capability to do infinitely more. This code acts as my learning process for understanding RAG and implementing it with Ollama, so I can query my files from anywhere without need for the internet. The use case for this technology is allowing businesses to create safe and local systems to query their own files without fear of big tech stealing their data.

**Future project enhancement:** Implement docker so the LLM can be run on a remote server and not suck up all the processing power of the user's computer.


#### Looking at the Paul Grahm text as an example, here are some example queries and responses:
Query: "In what context are aliens talked about in this document?"

Local LLM Response: "The context of alien discussion in this document revolves around the concept of invented vs discovered knowledge. It's used as an analogy to illustrate that even if advanced alien civilizations have access to Lisp, there might be other paths leading to discoveredness."

Query: "Who passed away in this document and why?"

Local LLM Response: "The author's mother passed away due to colon cancer, which caused a blood clot leading to a stroke."
