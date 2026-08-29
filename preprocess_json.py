# ==========================================
# ORIGINAL OLLAMA CODE (COMMENTED OUT FOR COMPARISON)
# ==========================================
# import requests
# import os
# import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# 
# def create_embedding(text_list):
#     # Send request to local Ollama server running on port 11434
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })
#     embedding = r.json()["embeddings"] 
#     return embedding
# 
# jsons = os.listdir("jsons") 
# my_dicts = []
# chunk_id = 0
# 
# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")
#     embeddings = create_embedding([c['text'] for c in content['chunks']])
#        
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk) 
# 
# df = pd.DataFrame.from_records(my_dicts)
# joblib.dump(df, 'embeddings.joblib')
# ==========================================


# ==========================================
# NEW GEMINI API CODE (WITH LINE-BY-LINE COMMENTS)
# ==========================================
import requests  # Import requests library to make HTTP REST API calls to Google Gemini
import os  # Import os module to read file paths and environment variables
import json  # Import json module to parse and write JSON data
import numpy as np  # Import numpy for numerical/vector array processing
import pandas as pd  # Import pandas to organize data in tabular DataFrame format
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity to find semantic matches
import joblib  # Import joblib to save and load Python objects on disk
import time  # Import time module to add delays during retries and backoff
from dotenv import load_dotenv  # Import load_dotenv to load settings from configuration files
from vector_db import RAGVectorDB  # Import vector database manager

# Load Gemini API key from local environment configuration file
load_dotenv()  

# Retrieve the Gemini API key from environment variables
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")  
# Check if API key is missing
if not api_key:  
    # Print error message if key was not found
    print("ERROR: GEMINI_API_KEY not found in environment.")  
    # Exit script execution immediately
    exit(1)  

# # Define the embedding function targeting Google Gemini embedContent REST API
def create_embedding(text_list):  
    sanitized_texts = [text if text.strip() else " " for text in text_list]  
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"  
    embeddings = []  
    
    for text in sanitized_texts:  
        payload = {  
            "model": "models/gemini-embedding-001",  
            "content": {  
                "parts": [{"text": text}]  
            }  
        }  
        max_retries = 10  
        for attempt in range(1, max_retries + 1):  
            try:  
                r = requests.post(url, json=payload, timeout=30.0)  
                if r.status_code == 429:  
                    print(f"Rate limited (429). Cooling down 5s... (Attempt {attempt}/{max_retries})", flush=True)  
                    time.sleep(5)  
                    continue  
                r.raise_for_status()  
                embeddings.append(r.json()["embedding"]["values"])  
                time.sleep(0.3)  # 0.3s delay between calls for smooth throughput
                break  
            except Exception as e:  
                if attempt == max_retries:  
                    raise e  
                time.sleep(3)  
                
    return embeddings  


def merge_chunks(chunks, group_size=15):
    """
    Merges small 2-3 second Whisper speech fragments into 45-60 second rich paragraph chunks
    for vastly improved RAG semantic retrieval accuracy and faster vector indexing.
    """
    merged = []
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i+group_size]
        combined_text = " ".join(c['text'].strip() for c in group if c.get('text'))
        if not combined_text:
            continue
        merged.append({
            "number": str(group[0].get("number", "")),
            "title": str(group[0].get("title", "")),
            "start": float(group[0].get("start", 0.0)),
            "end": float(group[-1].get("end", 0.0)),
            "text": combined_text
        })
    return merged


jsons = [f for f in os.listdir("jsons") if f.endswith(".json")]  
jsons.sort()

# Initialize list to hold chunks and their vector embeddings
my_dicts = []  

# Initialize ChromaDB vector database manager
vector_db = RAGVectorDB(db_path="./chroma_db", collection_name="teaching_assistant")

# Get list of video numbers already indexed in ChromaDB for incremental processing
existing_res = vector_db.collection.get(include=["metadatas"])
existing_numbers = set(str(m.get("number", "")) for m in existing_res.get("metadatas", []))
chunk_id = vector_db.count()

# Loop through each transcript file name
for json_file in jsons:  
    # Open the JSON transcript file explicitly with UTF-8 encoding
    with open(f"jsons/{json_file}", encoding="utf-8") as f:  
        content = json.load(f)  
    
    video_num = str(content.get("chunks", [{}])[0].get("number", ""))
    if video_num and video_num in existing_numbers:
        print(f"Skipping {json_file} (Video {video_num} already indexed in ChromaDB)", flush=True)
        continue

    # Merge micro-chunks into paragraph chunks
    paragraphs = merge_chunks(content.get("chunks", []), group_size=15)
    
    print(f"Processing {json_file} ({len(paragraphs)} paragraph chunks)...", flush=True)  
    
    # Enrich text chunks with video title context for higher quality semantic matching
    texts_to_embed = [
        f"Video {p.get('number', '')}: {p.get('title', '')} - {p['text']}"
        for p in paragraphs
    ]
    
    # Send all text chunks of this transcript to get embeddings
    embeddings = create_embedding(texts_to_embed)  

    ids = []
    chunk_metadatas = []
    documents = []
    chunk_embeddings = []
       
    # Loop through each merged paragraph chunk
    for i, p in enumerate(paragraphs):  
        c_id = f"chunk_{chunk_id}"
        p['chunk_id'] = chunk_id  
        p['embedding'] = embeddings[i]  
        
        ids.append(c_id)
        chunk_embeddings.append(embeddings[i])
        documents.append(p['text'])
        chunk_metadatas.append({
            "number": str(p.get("number", "")),
            "title": str(p.get("title", "")),
            "start": float(p.get("start", 0.0)),
            "end": float(p.get("end", 0.0))
        })
        
        chunk_id += 1  
        my_dicts.append(p)  

    # Upsert chunk batch into ChromaDB collection
    vector_db.add_chunks(
        ids=ids,
        embeddings=chunk_embeddings,
        metadatas=chunk_metadatas,
        documents=documents
    )
    existing_numbers.add(video_num)
    print(f"  Successfully indexed {len(ids)} paragraph chunks from {json_file}", flush=True)

# Save joblib backup if any dicts processed
if my_dicts:
    try:
        df = pd.DataFrame.from_records(my_dicts)  
        joblib.dump(df, 'embeddings_gemini.joblib')  
    except Exception:
        pass

print(f"Success! Total indexed chunks in ChromaDB: {vector_db.count()}", flush=True)
