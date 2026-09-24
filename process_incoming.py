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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

# 3. Setup Vector Store Retriever & Context Formatter
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    formatted_chunks = []
    for doc in docs:
        meta = doc.metadata
        formatted_chunks.append(
            f"Video Title: {meta.get('title', 'N/A')} (Video #{meta.get('number', 'N/A')})\n"
            f"Timestamps: {meta.get('start', 0.0)}s - {meta.get('end', 0.0)}s\n"
            f"Content: {doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)

# 4. Prompt Template & RAG Pipeline Setup
prompt_template = """You are an AI Teaching Assistant for web development video tutorials.
Answer the student's question based strictly on the retrieved context below.
Always cite the relevant video title, video number, and specific start/end timestamps in your response.

Retrieved Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Execute Query
def main():
    print("\n=======================================================")
    print(" RAG AI Teaching Assistant - Retrieval Pipeline Active")
    print("=======================================================\n")
    
    query = input("Ask a question about the course (or type 'exit' to quit): ")
    if query.strip() and query.strip().lower() not in ["exit", "quit"]:
        print("\nRetrieving relevant lecture segments and generating answer...")
        answer = rag_chain.invoke(query)
        print("\n--- Answer ---")
        print(answer)

if __name__ == "__main__":
    main()