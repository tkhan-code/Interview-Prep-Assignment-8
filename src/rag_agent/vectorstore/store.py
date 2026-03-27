"""Vector store management."""

import os
import hashlib
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings

# Use absolute import
from src.rag_agent.config import settings, EmbeddingFactory


class VectorStoreManager:
    
    def __init__(self):
        self.embedding_function = EmbeddingFactory.create()
        self._initialise()
    
    def _initialise(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="deep_learning_corpus",
            embedding_function=None
        )
    
    def _get_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
    
    def check_duplicate(self, chunk_text: str) -> bool:
        content_hash = self._get_content_hash(chunk_text)
        results = self.collection.get(where={"content_hash": content_hash})
        return len(results['ids']) > 0
    
    def ingest(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["chunk_text"]
            if self.check_duplicate(chunk_text):
                continue
            
            chunk_id = f"{chunk['metadata']['source']}_{i}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
            metadata = chunk["metadata"].copy()
            metadata["content_hash"] = self._get_content_hash(chunk_text)
            embedding = self.embedding_function.embed_documents([chunk_text])[0]
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(chunk_text)
        
        if ids:
            self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        
        return {"ingested": len(ids), "skipped_duplicates": len(chunks) - len(ids), "total_chunks": len(chunks)}
    
    def query(self, query_text: str, k: int = None) -> List[Dict[str, Any]]:
        if k is None:
            k = settings.retrieval_k
        
        query_embedding = self.embedding_function.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids'][0]):
                chunks.append({
                    "chunk_text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "id": doc_id
                })
        return chunks
    
    def list_documents(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for metadata in results['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
        return sorted(list(sources))
    
    def get_document_chunks(self, source: str) -> List[Dict[str, Any]]:
        results = self.collection.get(where={"source": source}, include=["documents", "metadatas"])
        chunks = []
        if results['ids']:
            for i in range(len(results['ids'])):
                chunks.append({
                    "chunk_text": results['documents'][i],
                    "metadata": results['metadatas'][i],
                    "id": results['ids'][i]
                })
        return chunks
