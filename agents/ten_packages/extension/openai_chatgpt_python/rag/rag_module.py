from pathlib import Path
from llama_index.core import SimpleDirectoryReader, StorageContext, Document
from llama_index.core.schema import ImageDocument, ImageNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
import qdrant_client
 
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, SimpleDirectoryReader
import qdrant_client
from pathlib import Path
from llama_index.core.indices import VectorStoreIndex

# Import modules
from langchain.chat_models import ChatOpenAI # changed from langchain_openai to langchain.chat_models
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import base64
import httpx
 
class RAGRetriever:
    def __init__(self, index):
        # Initialize the language model (e.g., GPT-4)
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.index = index
   
    def _get_tools(self):
        # Define tools if needed; leave empty here as no email is needed in this context
        return []
 
    def _get_retriever_engine(self, question):
        """Retrieve and organize text data from RAG results."""
        MAX_TOKENS = 50
        retrieval_text = question[:MAX_TOKENS]  # Use the question text directly
        retrieval_results = self.index.as_retriever(similarity_top_k=3).retrieve(retrieval_text)
 
        # Separate text nodes
        retrieved_texts = []
        for res_node in retrieval_results:
            if not isinstance(res_node.node, ImageNode):  # Only add text nodes
                text_content = f"Node ID: {res_node.node.id_}\nSimilarity Score: {res_node.score}\nText: {res_node.node.text[:500]}"
                retrieved_texts.append(text_content)
 
        return retrieved_texts
 
    def _build_prompt(self, retrieved_texts, question):
        """Build the prompt with text contexts."""
        prompt = "Based on the following information, provide a detailed answer to the question:\n\n"
       
        # Add question
        prompt += f"Question: {question}\n\n"
       
        # Add retrieved text contexts
        prompt += "Text Contexts:\n"
        for idx, text in enumerate(retrieved_texts, 1):
            prompt += f"Context {idx}:\n{text}\n\n"
       
        return prompt
 
    def get_final_response(self, question):
        """Retrieve context, build a structured prompt, and get the response from LLM."""
       
        # Step 1: Retrieve top-k context information from RAG
        MAX_TOKENS = 50
        retrieval_results = self.index.as_retriever(similarity_top_k=3).retrieve(question[:MAX_TOKENS])
 
        # Separate text and image nodes
        retrieved_texts = []
        retrieved_images = []
        print("=== Debugging: Retrieved Contexts ===")
        for idx, res_node in enumerate(retrieval_results, 1):
            if isinstance(res_node.node, ImageNode):
                # Collect image file paths
                retrieved_images.append(res_node.node.metadata["file_path"])
                print(f"[Image Context {idx}] Image Path: {res_node.node.metadata['file_path']}, Similarity Score: {res_node.score}")
            else:
                # Collect text with similarity score for context
                text_content = f"Node ID: {res_node.node.id_}\nSimilarity Score: {res_node.score}\nText: {res_node.node.text[:500]}"
                retrieved_texts.append(text_content)
                print(f"[Text Context {idx}] Node ID: {res_node.node.id_}, Similarity Score: {res_node.score}")
                print(f"Text: {res_node.node.text[:500]}\n")  # Truncate to show a preview of the text
 
        # Step 2: Build the structured prompt with both text and image contexts
        prompt = "Based on the following information, provide a detailed answer to the question:\n\n"
       
        # Add question
        prompt += f"Question: {question}\n\n"
       
        # Add retrieved text contexts
        prompt += "Text Contexts:\n"
        for idx, text in enumerate(retrieved_texts, 1):
            prompt += f"Context {idx}:\n{text}\n\n"
 
        # Add retrieved images, if any
        if retrieved_images:
            prompt += "Image References:\n"
            for idx, image_path in enumerate(retrieved_images, 1):
                prompt += f"Image {idx}: {image_path}\n"
        else:
            prompt += "No relevant images found.\n\n"
       
        # Step 3: Send the structured prompt to GPT-4 and get the response
        final_response = self.llm.invoke([{"role": "user", "content": prompt}])
       
        # Display the final answer
        print("Final Answer:", final_response.content)
        return final_response.content
 
 
def make_retriever():
    # Set up the RAG index (e.g., using multimodal vector store)
     
    # Initialize Qdrant client for local storage
    client = qdrant_client.QdrantClient(path="qdrant_mixed_db2")
 
    # Set up Qdrant vector stores for text
    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
 
    # Create a combined storage context with text store only
    storage_context = StorageContext.from_defaults(vector_store=text_store)
 
    # Load documents into the index (adjust the data path as needed)
    data_path = Path("./data")
    pdf_files = list(data_path.glob("*.pdf"))
    pdf_documents = [SimpleDirectoryReader(input_files=[pdf]).load_data() for pdf in pdf_files]
    all_documents = sum(pdf_documents, [])
 
    # Create the multimodal vector store index using Qdrant (for text only)
    index = VectorStoreIndex.from_documents(all_documents, storage_context=storage_context)
   
    # Example usage of RAGRetriever
    rag_retriever = RAGRetriever(index)

    return rag_retriever

if __name__ == "__main__":
    rag_retriever = make_retriever()
    question = "please describe cell Selection process"
    final_response = rag_retriever.get_final_response(question)
    print(final_response)


