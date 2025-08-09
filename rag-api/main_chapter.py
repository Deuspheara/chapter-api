#!/usr/bin/env python3
"""
Chapter-Based RAG System - Main Application
This is the main application file modified specifically for handling chapter-based JSON files.

This system is optimized for novels, stories, and serialized content where each chapter
is stored as a separate JSON file with rich metadata.
"""

import os
import asyncio
import logging
import time
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed  # python.import()
import signal  # python.import()

# Load environment variables from .env early
from dotenv import load_dotenv  # python.import()
load_dotenv()  # python.call()

# Core ML and NLP libraries
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, Range, MatchAny
)

# Web framework and API
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

# Our custom chapter processor
from chapter_processor import ChapterProcessor, ChapterRAGSystem

# Utilities
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChapterRAGConfig:
    """Configuration specifically optimized for chapter-based content."""
    
    # Model configurations - optimized for narrative content
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    # Qdrant settings
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME = "chapters"  # Renamed to be more specific
    VECTOR_SIZE = 1024
    
    # Chunking parameters optimized for narrative content
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))  # Slightly smaller for better narrative coherence
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # More overlap to preserve story flow
    
    # Retrieval parameters - optimized for speed
    INITIAL_RETRIEVAL_K = 20  # Reduced from 40 to 20 for faster reranking
    FINAL_RETRIEVAL_K = 8     # Optimized for story context
    
    # OpenAI settings
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    MAX_CONTEXT_LENGTH = 5000  # Larger context for story understanding

class EmbeddingService:
    """Text embedding service optimized for narrative content."""
    
    def __init__(self, config: ChapterRAGConfig):
        self.config = config
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Prefer Apple Silicon Metal (MPS) if available, otherwise CUDA, else CPU
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.model = self.model.to(device)
        logger.info(f"Embedding model loaded on {device}")
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for multiple texts with visible progress."""
        try:
            # For very large corpora, chunk the encode() calls to surface periodic progress logs,
            # while still using SentenceTransformer's internal batching per call.
            chunk_size_for_progress = 2000  # controls how often we log; not the model batch_size
            all_embeddings = []
            total = len(texts)
            processed = 0

            for start in range(0, total, chunk_size_for_progress):
                end = min(start + chunk_size_for_progress, total)
                subtexts = texts[start:end]
                embeddings = self.model.encode(
                    subtexts,
                    batch_size=64,               # model micro-batch; tune if OOM
                    normalize_embeddings=True,
                    show_progress_bar=show_progress,  # show tqdm for each encode slice
                    convert_to_tensor=False
                )
                all_embeddings.append(embeddings)
                processed = end
                # Log every slice completion
                logger.info(f"Embedding progress: {processed}/{total} texts")

            # Concatenate to a single numpy array
            import numpy as _np
            embeddings = _np.vstack(all_embeddings) if all_embeddings else _np.empty((0, 0))
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.model.encode([query], normalize_embeddings=True)[0]

class ChapterVectorStore:
    """Vector store optimized for chapter-based content."""
    
    def __init__(self, config: ChapterRAGConfig):
        self.config = config
        # Qdrant client with extended timeout
        self.client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            timeout=120.0,  # Increased HTTP timeout per request for larger batches
            prefer_grpc=False,  # Force REST API to avoid gRPC issues
        )
        
        # Log version compatibility info for debugging
        try:
            import qdrant_client
            logger.info(f"Qdrant client version: {getattr(qdrant_client, '__version__', 'Unknown')}")
            
            # Retry connection with backoff
            max_retries = 10
            retry_delay = 2
            for attempt in range(max_retries):
                try:
                    server_info = self.client.get_collections()
                    logger.info(f"Successfully connected to Qdrant server")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 30)  # Exponential backoff, max 30 seconds
                    else:
                        logger.error(f"Failed to connect to Qdrant after {max_retries} attempts: {e}")
                        raise
                        
        except Exception as e:
            logger.warning(f"Could not get server info: {e}")
        self._setup_collection()
        logger.info("Chapter vector store initialized")
    
    def _setup_collection(self):
        """Initialize the Qdrant collection with chapter-specific settings."""
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                collections = self.client.get_collections().collections
                collection_names = [col.name for col in collections]
                
                if self.config.COLLECTION_NAME in collection_names:
                    logger.info(f"Collection {self.config.COLLECTION_NAME} already exists")
                    return
                
                self.client.create_collection(
                    collection_name=self.config.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.config.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.config.COLLECTION_NAME}")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Collection setup attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 10)  # Exponential backoff, max 10 seconds
                else:
                    logger.error(f"Error setting up collection after {max_retries} attempts: {str(e)}")
                    raise
    
    def _batched(self, iterable, n):
        """Yield successive n-sized batches from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    def _build_points(self, chunk_batch: List[Dict], emb_batch: np.ndarray) -> List[PointStruct]:
        """Build PointStruct list for a batch."""
        import hashlib
        points = []
        for chunk, embedding in zip(chunk_batch, emb_batch):
            # Convert string ID to UUID for Qdrant v1.15+ compatibility
            string_id = chunk['id']
            # Create a deterministic UUID from the string ID
            uuid_bytes = hashlib.md5(string_id.encode()).digest()
            uuid_str = str(uuid.UUID(bytes=uuid_bytes))
            
            points.append(
                PointStruct(
                    id=uuid_str,
                    vector=embedding.tolist(),
                    payload={
                        'text': chunk['text'],
                        'original_id': string_id,  # Keep original ID in payload
                        **chunk['metadata']
                    }
                )
            )
        return points

    def _drain_some(self, futures, drain_all: bool = False):
        """Collect completed futures. If drain_all, wait for all."""
        completed = []
        if drain_all:
            for fut in as_completed(futures):
                completed.append(fut.result())
            return completed, []
        else:
            still_pending = []
            for fut in futures:
                if fut.done():
                    completed.append(fut.result())
                else:
                    still_pending.append(fut)
            return completed, still_pending

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray, batch_size: int = 1000, workers: int = 4):
        """
        Add chapter chunks using the standard Qdrant client API v1.8.0.
        Uses the upsert method with PointStruct objects.
        """
        try:
            total = len(chunks)
            if total != len(embeddings):
                raise ValueError(f"Chunks and embeddings size mismatch: {total} vs {len(embeddings)}")

            added = 0
            first_batch_logged = False

            # Process in batches sequentially to avoid overwhelming the server
            for chunk_batch, emb_batch in zip(self._batched(chunks, batch_size), self._batched(embeddings, batch_size)):
                points = self._build_points(chunk_batch, emb_batch)

                # One-time debug snapshot of the first batch
                if not first_batch_logged:
                    try:
                        sample_point = points[0] if points else None
                        logger.info(
                            f"Qdrant upsert first batch debug: count={len(points)}, "
                            f"point_id={sample_point.id if sample_point else None}, "
                            f"vector_len={len(sample_point.vector) if sample_point else 0}"
                        )
                    except Exception as _e:
                        logger.debug(f"Debug log for first batch failed: {str(_e)}")
                    first_batch_logged = True

                # Use the standard client upsert method
                self.client.upsert(
                    collection_name=self.config.COLLECTION_NAME,
                    points=points,
                    wait=True
                )
                
                added += len(points)
                logger.info(f"Upsert progress: {min(added, total)}/{total} points")

            logger.info(f"Finished upserting {added} points with batch_size={batch_size}")

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, limit: int = 10, 
               filter_conditions: Optional[Filter] = None) -> List[Dict]:
        """Search for similar chapter chunks."""
        try:
            search_result = self.client.search(
                collection_name=self.config.COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                query_filter=filter_conditions,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for hit in search_result:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'text': hit.payload['text'],
                    'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
                })
            
            logger.info(f"Found {len(results)} similar chapter chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

class ChapterReranker:
    """Reranker optimized for narrative content."""
    
    def __init__(self, config: ChapterRAGConfig):
        self.config = config
        logger.info(f"Loading reranker model: {config.RERANKER_MODEL}")
        self.model = CrossEncoder(config.RERANKER_MODEL)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if hasattr(self.model, 'model'):
            self.model.model = self.model.model.to(device)
        
        logger.info(f"Reranker model loaded on {device}")
    
    def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Rerank documents for narrative relevance."""
        try:
            if len(documents) <= top_k:
                return documents
            
            query_doc_pairs = [[query, doc['text']] for doc in documents]
            scores = self.model.predict(query_doc_pairs)
            
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
                # Weight narrative coherence higher
                doc['combined_score'] = 0.2 * doc['score'] + 0.8 * score
            
            reranked_docs = sorted(documents, key=lambda x: x['combined_score'], reverse=True)[:top_k]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {top_k}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            return documents[:top_k]

class MainChapterRAGSystem:
    """Main RAG system specifically designed for chapter-based content."""
    
    def __init__(self, openai_api_key: str):
        self.config = ChapterRAGConfig()
        
        # Initialize OpenAI
        openai.api_key = openai_api_key
        
        # Initialize components
        self.embedding_service = EmbeddingService(self.config)
        self.vector_store = ChapterVectorStore(self.config)
        self.reranker = ChapterReranker(self.config)
        self.chapter_processor = ChapterProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        
        # Initialize the specialized chapter RAG system
        self.chapter_rag = ChapterRAGSystem(self, self.chapter_processor)
        
        logger.info("Chapter-based RAG system fully initialized")
    
    def index_chapters(self, directory_path: str, file_pattern: str = "chapter-*.json") -> int:
        """Index chapters from JSON files."""
        return self.chapter_rag.index_chapters(directory_path, file_pattern)
    
    def query_chapters(self, question: str, 
                      specific_chapters: Optional[List[int]] = None,
                      use_reranking: bool = True) -> Dict:
        """Query the chapter-based system."""
        return self.chapter_rag.query_with_chapter_context(
            question, 
            specific_chapters, 
            use_reranking
        )
    
    def get_chapter_summary(self) -> Dict:
        """Get a summary of all indexed chapters."""
        chapters = self.chapter_rag.get_chapter_list()
        
        if not chapters:
            return {"message": "No chapters indexed yet."}
        
        total_chapters = len(chapters)
        total_words = sum(ch.get('word_count', 0) for ch in chapters)
        
        return {
            "total_chapters": total_chapters,
            "total_words": total_words,
            "average_words_per_chapter": total_words // total_chapters if total_chapters > 0 else 0,
            "chapter_range": f"Chapter {min(ch['number'] for ch in chapters)} - Chapter {max(ch['number'] for ch in chapters)}",
            "chapters": chapters
        }

# API Models
class ChapterIndexRequest(BaseModel):
    directory_path: str
    file_pattern: str = "chapter-*.json"

class ChapterQueryRequest(BaseModel):
    question: str
    # Accept list OR legacy dict shapes; validate/normalize manually in handler
    specific_chapters: Optional[object] = None
    use_reranking: bool = True
    fast_mode: bool = False  # Skip reranking for speed

class ChapterQueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float
    processing_time: float
    chapter_summary: Optional[str] = None
    chapters_referenced: Optional[List[int]] = None

# FastAPI Application
app = FastAPI(
    title="Chapter-Based RAG System",
    description="A RAG system specialized for novels, stories, and chapter-based content",
    version="1.0.0"
)

# Global system instance
chapter_rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chapter RAG system on startup."""
    global chapter_rag_system
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise RuntimeError("OpenAI API key is required")
    
    chapter_rag_system = MainChapterRAGSystem(openai_api_key)
    logger.info("Chapter RAG system started successfully")

@app.post("/index-chapters")
async def index_chapters(request: ChapterIndexRequest):
    """Index chapter JSON files from a directory."""
    try:
        chunk_count = chapter_rag_system.index_chapters(
            request.directory_path,
            request.file_pattern
        )
        
        # Get chapter summary after indexing
        summary = chapter_rag_system.get_chapter_summary()
        
        return {
            "status": "success",
            "message": f"Successfully indexed {chunk_count} chunks",
            "chunk_count": chunk_count,
            "chapter_summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=ChapterQueryResponse)
async def query_chapters(request: ChapterQueryRequest):
    """Query the chapter-based RAG system."""
    try:
        # Normalize specific_chapters to a plain list[int] if possible
        normalized_specific = None
        raw_specific = request.specific_chapters
        try:
            if raw_specific is None:
                normalized_specific = None
            elif isinstance(raw_specific, dict):
                # Accept {"any":{"values":[...]}} or {"values":[...]}
                any_obj = raw_specific.get("any")
                if isinstance(any_obj, dict) and isinstance(any_obj.get("values"), list):
                    normalized_specific = any_obj["values"]
                elif isinstance(raw_specific.get("values"), list):
                    normalized_specific = raw_specific["values"]
                else:
                    normalized_specific = None
            elif isinstance(raw_specific, (list, tuple)):
                normalized_specific = list(raw_specific)
            else:
                normalized_specific = None

            # Cast all to int and drop non-castable
            if isinstance(normalized_specific, list):
                cleaned = []
                for v in normalized_specific:
                    try:
                        cleaned.append(int(v))
                    except Exception:
                        continue
                normalized_specific = cleaned if cleaned else None
        except Exception as _e:
            logger.debug(f"specific_chapters normalization failed: {str(_e)}")
            normalized_specific = None

        # Use fast mode for recap queries to skip expensive reranking
        use_reranking = request.use_reranking and not request.fast_mode
        
        result = chapter_rag_system.query_chapters(
            request.question,
            normalized_specific,
            use_reranking
        )
        return ChapterQueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chapters")
async def get_chapters():
    """Get information about all indexed chapters."""
    try:
        return chapter_rag_system.get_chapter_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_type": "chapter_rag",
        "version": "1.0.0"
    }

@app.get("/stats")
async def get_stats():
    """Get detailed system statistics."""
    try:
        if chapter_rag_system:
            collection_info = chapter_rag_system.vector_store.client.get_collection(
                chapter_rag_system.config.COLLECTION_NAME
            )
            
            chapter_summary = chapter_rag_system.get_chapter_summary()
            
            return {
                "documents_indexed": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "collection_name": chapter_rag_system.config.COLLECTION_NAME,
                "chapter_info": chapter_summary,
                "config": {
                    "chunk_size": chapter_rag_system.config.CHUNK_SIZE,
                    "chunk_overlap": chapter_rag_system.config.CHUNK_OVERLAP,
                    "embedding_model": chapter_rag_system.config.EMBEDDING_MODEL,
                    "reranker_model": chapter_rag_system.config.RERANKER_MODEL
                }
            }
        return {"status": "system not initialized"}
    except Exception as e:
        return {"error": str(e)}

# Example queries endpoint for demo purposes
@app.get("/example-queries")
async def get_example_queries():
    """Get example queries that work well with chapter-based content."""
    return {
        "character_questions": [
            "Who is Zhou Mingrui and what happened to him?",
            "Describe Klein Moretti's family situation.",
            "What characters appear in the first chapter?"
        ],
        "plot_questions": [
            "What happened when Zhou Mingrui first woke up?",
            "How did Zhou Mingrui realize he had transmigrated?",
            "What mysterious elements appear in Chapter 1?"
        ],
        "setting_questions": [
            "Describe the room Zhou Mingrui woke up in.",
            "What world or setting is this story in?",
            "What objects were on the desk?"
        ],
        "analysis_questions": [
            "What themes are introduced in the opening chapter?",
            "How does the author establish the mysterious atmosphere?",
            "What clues suggest this is a transmigration story?"
        ]
    }

if __name__ == "__main__":
    # Run the development server with proper signal handling (SIGINT/SIGTERM).
    uvicorn.run(
        "main_chapter:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True
    )