# RAG AI Teaching Assistant

A state-of-the-art **Retrieval-Augmented Generation (RAG)** system built with **LangChain**, **Google Gemini**, **ChromaDB**, and **OpenAI Whisper**. 

This system indexes course video lectures into a persistent vector store and deploys an autonomous **ReAct AI Agent** that dynamically retrieves relevant lecture passages and answers student questions with precise video titles, video numbers, and exact start/end timestamp citations.

---

---

## Tech Stack

* **Frameworks:** LangChain, LangGraph
* **Language Model (LLM):** Google Gemini 2.5 Flash (`gemini-2.5-flash`)
* **Embedding Model:** Google Gemini Embedding (`gemini-embedding-001`)
* **Vector Database:** ChromaDB (`chromadb`)
* **Speech Recognition (ASR):** OpenAI Whisper (`whisper` `large-v2`)
* **Media Processing:** FFmpeg
* **Language & Runtime:** Python 3.9+
* **Environment Management:** `python-dotenv`

---

## Project Structure

```text
RAG-AI-Teaching-Assistant/
├── chroma_db/                # Persistent ChromaDB vector database directory
├── jsons/                    # Preprocessed JSON transcript files with timestamps
├── video_to_mp3.py           # Step 1: Extracts MP3 audio from raw course videos
├── mp3_to_json.py             # Step 2: Speech-to-text transcription via OpenAI Whisper
├── preprocess_json.py        # Step 3: Batch chunking & ChromaDB vector indexer
├── process_incoming.py       # Step 4: Main Agentic RAG script (LangChain + LangGraph Agent)
├── .env                      # API Key configuration file
└── Readme.md                 # Project documentation
```

---
