import os
import chromadb

class RAGVectorDB:
    def __init__(self, db_path="./chroma_db", collection_name="teaching_assistant"):
        """
        Initializes persistent ChromaDB client and collection.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, ids, embeddings, metadatas, documents):
        """
        Upserts transcript chunks with embeddings and metadata into ChromaDB collection.
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query(self, query_embedding, n_results=5, where_filter=None):
        """
        Queries ChromaDB collection using a vector embedding and optional metadata filter.
        Returns top n_results matching documents and metadata.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)
        return results

    def reset_collection(self):
        """
        Deletes and recreates the collection for fresh re-indexing.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self):
        """
        Returns total number of items indexed in the collection.
        """
        return self.collection.count()
