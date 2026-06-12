#Import PDF loader to read the pdf file 
from langchain_community.document_loaders import PyPDFLoader

#load the pdf file 
loader = PyPDFLoader("sample_txt.pdf")

#Read pdf pages
documents = loader.load()

#check how many pages were loaded 
print("Page loaded:", len(documents))

#Split large text into smaller chunks 
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size = 500,

    chunk_overlap = 50

)

chunks = splitter.split_documents(documents)

print("chunks created:", len(chunks))

#convert text chunks into embeddings 
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded")

#Store chunk embeddings in FAISS 
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector database created")

# Retriever searches the FAISS database and returns the most relevant chunks
retriever = vectorstore.as_retriever(

    search_kwargs={"k": 3}
)

print("Retriever created")

#load FLAN-T5 model and tokenizer
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("FLAN-T5 loaded")

from typing import TypedDict

#Shared data used by all langGraph nodes
class GraphState(TypedDict):

    question: str

    route : str

    context : str

    answer : str 

# Decide which route the question should take
def router(state: GraphState):

    question = state["question"].lower()

    # Questions related to the PDF
    pdf_keywords = [
        "rag",
        "rlhf",
        "embeddings",
        "langchain",
        "vector database",
        "llm",
        "artificial intelligence"
    ]

    for keyword in pdf_keywords:
        if keyword in question:
            return {"route": "pdf"}

    # Everything else goes to general generation
    return {"route": "general"}

# Retrieve relevant information from the pdf 
def retrieve(state: GraphState):

    question = state["question"]

    documents = retriever.invoke(question)

    for doc in documents:
        print(doc.page_content)
        print("-" * 50)

    context = "\n".join([doc.page_content for doc in documents[:3]])

    return {"context": context}

# Generate answer using FLAN-T5
def generate(state: GraphState):

    question = state["question"]
    context = state["context"]

    if context:

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    else:

        prompt = f"""
Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {"answer": answer}

from langgraph.graph import StateGraph, START, END 

graph = StateGraph(GraphState)

graph.add_node("router", router)

graph.add_node("retrieve", retrieve)

graph.add_node("generate", generate)

#Start the workflow from router
graph.add_edge(START, "router")

#Decide where to go after router 
def route_decision(state: GraphState):

    return state["route"]

graph.add_conditional_edges(

    "router",

    route_decision,
    {
        "pdf": "retrieve",
        "general": "generate"

    }
)

#after retrieval, generate the answer 
graph.add_edge("retrieve", "generate")

#End after generation
graph.add_edge("generate", END)

#compile the graph
app = graph.compile()


