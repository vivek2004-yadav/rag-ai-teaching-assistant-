import os
import json
import glob
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# Load API keys from .env
load_dotenv()

# Set GOOGLE_API_KEY for LangChain Google GenAI integration
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")
    exit(1)

# 1. Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
db_path = "./chroma_db"

# 2. Connect to existing Chroma DB or build from JSONs using RecursiveCharacterTextSplitter
if os.path.exists(db_path):
    print("Loading existing ChromaDB from ./chroma_db...")
    vector_store = Chroma(
        persist_directory=db_path,
        collection_name="teaching_assistant",
        embedding_function=embeddings
    )
else:
    print("ChromaDB not found. Loading and splitting JSON transcripts using RecursiveCharacterTextSplitter...")
    raw_docs = []
    json_files = glob.glob("./jsons/*.json")

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks = data.get("chunks", []) if isinstance(data, dict) else data
            for chunk in chunks:
                if isinstance(chunk, dict):
                    doc = Document(
                        page_content=chunk.get("text", ""),
                        metadata={
                            "title": chunk.get("title", ""),
                            "number": str(chunk.get("number", "")),
                            "start": float(chunk.get("start", 0.0)),
                            "end": float(chunk.get("end", 0.0))
                        }
                    )
                    raw_docs.append(doc)

    print(f"Loaded {len(raw_docs)} raw transcript fragments. Splitting with RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(raw_docs)

    print(f"Indexing {len(split_docs)} passage chunks into ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="teaching_assistant"
    )

# 3. Agent Tool (retriever_tool)
@tool
def retriever_tool(query: str) -> str:
    """This tool retrieves relevant video subtitle chunks, including video titles and timestamps for web development course questions."""
    print("Tool Called:", query)
    retrieved_docs = vector_store.similarity_search(query=query, k=4)
    
    context = ""
    for doc in retrieved_docs:
        meta = doc.metadata
        context += (
            f"Video Title: {meta.get('title')} (Video #{meta.get('number')})\n"
            f"Timestamps: {meta.get('start')}s - {meta.get('end')}s\n"
            f"Content: {doc.page_content}\n\n"
        )
    return context

# 4. LLM & System Prompt Setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

System_Prompt = """You are a helpful assistant that answers questions using retrieved context from web development video tutorials.
ALWAYS use the `retriever_tool` tool for questions requiring external knowledge.
Include video titles and timestamps in your answer."""

memory = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[retriever_tool],
    prompt=System_Prompt,
    checkpointer=memory
)

# 5. Execute Query
query = input("\nAsk a question about the course: ")
if query.strip():
    config = {"configurable": {"thread_id": "1"}}
    response = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
    result = response["messages"][-1].content

    print("\n--- Agent Answer ---")
    print(result)