import os
import json
import glob
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables (.env)
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit(1)

def preprocess_and_index():
    json_dir = "./jsons"
    db_path = "./chroma_db"
    
    print("Loading transcript JSON files from ./jsons...")
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    raw_docs = []
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

    print(f"Loaded {len(raw_docs)} transcript fragments. Splitting using RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(raw_docs)

    print(f"Creating vector embeddings for {len(split_docs)} chunks and saving to ./chroma_db...")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="teaching_assistant"
    )
    
    print("Indexing complete! Persistent database saved at ./chroma_db")

if __name__ == "__main__":
    preprocess_and_index()
