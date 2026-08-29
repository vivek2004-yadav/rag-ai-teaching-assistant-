# ==========================================
# ORIGINAL OLLAMA CODE (COMMENTED OUT FOR COMPARISON)
# ==========================================
# import pandas as pd 
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np 
# import joblib 
# import requests
# 
# def create_embedding(text_list):
#     # Query local Ollama server running on port 11434
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })
#     embedding = r.json()["embeddings"] 
#     return embedding
# 
# def inference(prompt):
#     # Generate answer via local llama3.2 model in Ollama
#     r = requests.post("http://localhost:11434/api/generate", json={
#         "model": "llama3.2",
#         "prompt": prompt,
#         "stream": False
#     })
#     response = r.json()
#     print(response)
#     return response
# 
# df = joblib.load('embeddings.joblib')
# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0] 
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# top_results = 5
# max_indx = similarities.argsort()[::-1][0:top_results]
# new_df = df.loc[max_indx] 
# 
# prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:
# {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
# ---------------------------------
# "{incoming_query}"
# User asked this question related to the video chunks, you have to answer in a human way...
# '''
# with open("prompt.txt", "w") as f:
#     f.write(prompt)
# response = inference(prompt)["response"]
# print(response)
# with open("response.txt", "w") as f:
#     f.write(response)
# ==========================================


# ==========================================
# NEW GEMINI API CODE (WITH LINE-BY-LINE COMMENTS)
# ==========================================
import pandas as pd  # Import pandas to organize data in tabular DataFrame format
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity to find semantic matches
import numpy as np  # Import numpy for numerical/vector array processing
import joblib  # Import joblib to save and load Python objects on disk
import requests  # Import requests library to make HTTP REST API calls to Google Gemini
import os  # Import os module to check file paths and environment variables
import json  # Import json module to handle json formatting
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

# Define the function to generate vector embedding for user queries via Gemini API
def create_embedding(text_list):  
    # Construct the Gemini REST API URL for query embedding generation
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"  
    # Prepare the payload with the target embedding model and the input text
    payload = {  
        "model": "models/gemini-embedding-001",  
        "content": {  
            "parts": [{"text": text_list[0]}]  
        }  
    }  
    # Send POST request to Google Gemini API
    r = requests.post(url, json=payload, timeout=30.0)  
    # Raise an error if the status code indicates a failure
    r.raise_for_status()  
    # Extract the vector float values array from the JSON response
    embedding = r.json()["embedding"]["values"]  
    # Return the embedding vector wrapped in a list to match original format
    return [embedding]  

# Define inference function to query the Google Gemini LLM with prompt context
def inference(prompt):  
    # Import time module to add delays during retries and backoff
    import time  
    # Define fallback model list to ensure system remains functional on transient errors
    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]  
    # Initialize placeholder for the last exception raised
    last_err = None  
    # Loop over the prioritized models
    for model in models:  
        # Construct the Gemini REST API URL for content generation
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"  
        # Build the payload structure required by Google's API
        payload = {  
            "contents": [  
                {  
                    "parts": [  
                        {"text": prompt}  
                    ]  
                }  
            ]  
        }  
        # Set max retry limit for this specific model
        max_retries = 3  
        # Try to call the API, retrying up to max_retries times
        for attempt in range(1, max_retries + 1):  
            # Try block to catch status codes and exceptions
            try:  
                # Send the POST request to Google Gemini API
                r = requests.post(url, json=payload, timeout=60.0)  
                # Check if rate limited (429) or server is overloaded (503)
                if r.status_code in [429, 503]:  
                    # Notify user that retry is active and delay is triggered
                    print(f"Warning: Server returned {r.status_code} for {model}. Retrying in 5 seconds... (Attempt {attempt}/{max_retries})")  
                    # Sleep execution for 5 seconds to cool down API rate limits
                    time.sleep(5)  
                    # Continue loop to next retry attempt
                    continue  
                # Raise an error if the status code indicates a failure
                r.raise_for_status()  
                # Extract response text generated by the model
                response_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]  
                # Return the response dictionary formatted like the original
                return {"response": response_text}  
            # Catch block for exceptions
            except Exception as e:  
                # Save the exception to raise if all retries and models fail
                last_err = e  
                # Check if this was the last allowed retry attempt for this model
                if attempt == max_retries:  
                    # Print warning indicating attempt to fall back to another model
                    print(f"Warning: Failed with {model} after {max_retries} attempts. Trying next model...")  
                # Otherwise, wait briefly
                else:  
                    # Sleep execution for 2 seconds before next attempt
                    time.sleep(2)  
    # Raise the last caught exception if all models failed
    raise last_err  

# Check Vector Database availability
db_path = "./chroma_db"

if not os.path.exists(db_path):  
    print(f"ERROR: Vector Database ('{db_path}') was not found. Please run 'python preprocess_json.py' to initialize ChromaDB.")  
    exit(1)  

# Prompt the user to input their question in terminal
incoming_query = input("Ask a Question: ")  
# Generate the vector embedding for user query
question_embedding = create_embedding([incoming_query])[0]  

matched_chunks = []

# Perform search via ChromaDB vector store
vector_db = RAGVectorDB(db_path=db_path, collection_name="teaching_assistant")
query_res = vector_db.query(query_embedding=question_embedding, n_results=10)

docs = query_res.get("documents", [[]])[0]
metas = query_res.get("metadatas", [[]])[0]

for i in range(len(docs)):
    meta = metas[i]
    matched_chunks.append({
        "title": meta.get("title", ""),
        "number": meta.get("number", ""),
        "start": meta.get("start", 0.0),
        "end": meta.get("end", 0.0),
        "text": docs[i]
    })

# Construct prompt template with retrieved context chunks
prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{json.dumps(matched_chunks, ensure_ascii=False)}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a clear and natural way (dont mention the above raw data format, it is just for your reference) where and how much content is taught in which video (including video title, number, and timestamps). Respond in English. If user asks an unrelated question, inform them that you can only answer questions related to the course.
'''  
# Open prompt.txt explicitly with UTF-8 encoding
with open("prompt.txt", "w", encoding="utf-8") as f:  
    # Write full prompt context to disk
    f.write(prompt)  

# Send the constructed prompt context to Google Gemini API
response = inference(prompt)["response"]  
# Print header for model response block
print("\n--- Gemini AI Teacher Response ---")  
# Print the model's generated answer to stdout
print(response)  
# Print footer for model response block
print("-----------------------------------")  

# Open response.txt explicitly with UTF-8 encoding
with open("response.txt", "w", encoding="utf-8") as f:  
    # Save final response output to disk
    f.write(response)  
  
