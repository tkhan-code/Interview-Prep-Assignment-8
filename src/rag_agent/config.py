"""Configuration management for RAG Agent."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings."""
    
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))


settings = Settings()


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create():
        if settings.llm_provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY not set")
            from langchain_groq import ChatGroq
            return ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, temperature=0.7)
        else:
            raise ValueError(f"Unknown provider: {settings.llm_provider}")


class EmbeddingFactory:
    """Factory for creating embedding models."""
    
    @staticmethod
    def create():
        # Simple embedding without any external dependencies
        class SimpleEmbeddings:
            def __init__(self):
                self.dimension = 384
            
            def embed_documents(self, texts):
                import hashlib
                embeddings = []
                for text in texts:
                    emb = [0.0] * self.dimension
                    words = text.lower().split()[:100]
                    for word in words:
                        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
                        idx = hash_val % self.dimension
                        emb[idx] += 1.0
                    # Normalize
                    norm = sum(x*x for x in emb)**0.5
                    if norm > 0:
                        emb = [x/norm for x in emb]
                    embeddings.append(emb)
                return embeddings
            
            def embed_query(self, text):
                return self.embed_documents([text])[0]
        
        return SimpleEmbeddings()
