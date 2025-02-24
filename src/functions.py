from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
from qdrant_client import QdrantClient
import numpy as np
from langgraph.graph import END, StateGraph
from prompt import route_chain,rag_chain
from models_llm import llm_embed_small
import os

qdrant_client = QdrantClient(
    url=os.getenv('url'), 
    api_key=os.getenv('apikey'),
)
def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    """
    Retrieve a chunk from the vectorstore based on its index in the metadata.
    
    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    target_index (int): The index of the chunk to retrieve.
    
    Returns:
    Optional[Document]: The retrieved chunk as a Document object, or None if not found.
    """
    # This is a simplified version. In practice, you might need a more efficient method
    # to retrieve chunks by index, depending on your vectorstore implementation.
    query_vector = np.zeros(1536).tolist()

    all_docs = vectorstore.search(
    collection_name="model_v2",
    query_vector=query_vector,
    limit=vectorstore.get_collection("model_v2").points_count  
    )

    for doc in all_docs:
        if doc.payload.get("index") == target_index:
            return doc
    return None
def index_retrieve_with_context(index,vectorstore, num_neighbors: int) -> str:

    current_index = index
    start_index = max(1, current_index - num_neighbors)
    end_index = current_index + num_neighbors + 1

    # Retrieve all chunks in the range
    neighbor_chunks = []
    for i in range(start_index, end_index):
        try:
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            neighbor_chunks.append(neighbor_chunk)
        except:
            pass

    text_chunks = []
    for chunk in neighbor_chunks:
        text_chunks.append(chunk.payload["text"])

    text_to_retrieve = " ".join(text_chunks)

    return text_to_retrieve
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents 
    """
    question : str
    generation : str
    documents : str
def route_question(state):
    """
    Route question to otros,process or RAG 

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = route_chain.invoke({"question": question})
    try:
        result = source['answer']
        return result
    except:
        result = "otros"
        return result
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE from Vector Store DB---")
    
    question = state["question"]

    chunks_query_retriever = qdrant_client.search(
    collection_name="model_v2",
    query_vector=llm_embed_small.embed_query(question),
    limit=1
    )

    doc_index = chunks_query_retriever[0].payload['index']

    filtered_docs = index_retrieve_with_context(
        index = doc_index,
        vectorstore = qdrant_client,
        num_neighbors = 1
        )
    

    print(filtered_docs)
    return {"documents": filtered_docs, "question": question,}
def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE Answer---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    print(generation)
    return {"documents": documents, "question": question, "generation": generation}

def answers_predefined(state):
    question = state['question']
    if "documents" in state:
        documents = state['documents']
        if documents == None:
            return {'question':question,'generation':"Actualmente solo puedo con los datos mencionados en el documento de prueba","documents":""}
        else:
            return {'question':question,'generation':"En el momento no cuento con el contexto necesario para responder tu pregunta","documents":""}
    else:
        return {'question':question,'generation':"Actualmente solo puedo ayudarte con tu consulta","documents":""}
    
def initial(state):
    return "route_question"

def retrieve_to_generate(state):
    return "generate"

def retrieve_process_to_generate(state):
    return "generate_process"
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("predefined", answers_predefined) # answers predefined
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("generate", generate) # generate 
workflow.set_conditional_entry_point(
    route_question,
    {
        "data": "retrieve",
        "otros": "predefined",
    },
)
workflow.add_conditional_edges(
    "retrieve", # start: node
    retrieve_to_generate, # defined function
    {
        "generate": "generate", #returns of the function
    }
)

app_langgraph = workflow.compile()
def flujo(input):
    for output in app_langgraph.stream(input):
        for key, value in output.items():
            print(f"Finished running: {key}:")
    return value




def flujo(input):
    for output in app_langgraph.stream(input):
        for key, value in output.items():
            print(f"Finished running: {key}:")
    return value["generation"]

# inputs = {"question": "¿Por qué las aseguradoras que adopten IA tendrán ventaja competitiva?"}

# flujo(inputs)