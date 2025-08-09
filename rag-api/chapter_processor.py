#!/usr/bin/env python3
"""
Chapter-Specific Document Processor for RAG System
This module handles JSON-formatted chapter files for novels/books and processes them for RAG.

This processor is specifically designed for your chapter format where each file contains
structured chapter data including metadata, content, and paragraph information.
"""

import json
import logging
import time
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import re
import numpy as np

logger = logging.getLogger(__name__)

class ChapterProcessor:
    """
    Specialized processor for handling chapter-based JSON files.
    
    This processor understands your specific chapter format and extracts
    meaningful information while preserving important context about
    which chapter each piece of information comes from.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter optimized for narrative content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Prioritize paragraph and sentence boundaries for narrative text
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        logger.info("Chapter processor initialized")
    
    def load_chapter_files(self, directory_path: str, file_pattern: str = "chapter-*.json") -> List[Dict]:
        """
        Load all chapter files from a directory.
        
        Args:
            directory_path: Path to directory containing chapter JSON files
            file_pattern: Glob pattern for chapter files (default: "chapter-*.json")
            
        Returns:
            List of parsed chapter data dictionaries
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        chapter_files = sorted(directory.glob(file_pattern))
        if not chapter_files:
            raise ValueError(f"No chapter files found matching pattern: {file_pattern}")
        
        chapters = []
        
        for file_path in chapter_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chapter_data = json.load(f)
                    
                # Validate required fields
                if not self._validate_chapter_data(chapter_data):
                    logger.warning(f"Skipping invalid chapter file: {file_path}")
                    continue
                
                # Add file information to chapter data
                chapter_data['source_file'] = str(file_path)
                chapter_data['filename'] = file_path.name
                chapters.append(chapter_data)
                
                logger.info(f"Loaded chapter {chapter_data.get('number', '?')}: {chapter_data.get('title', 'Unknown')}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file {file_path}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading chapter file {file_path}: {e}")
                continue
        
        if not chapters:
            raise ValueError("No valid chapter files could be loaded")
        
        # Sort chapters by number to ensure proper order
        chapters.sort(key=lambda x: x.get('number', 0))
        
        logger.info(f"Successfully loaded {len(chapters)} chapters")
        return chapters
    
    def _validate_chapter_data(self, chapter_data: Dict) -> bool:
        """
        Validate that chapter data contains required fields.
        
        Args:
            chapter_data: Dictionary containing chapter information
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['title', 'content']
        optional_but_useful = ['number', 'word_count', 'character_count']
        
        # Check required fields
        for field in required_fields:
            if field not in chapter_data or not chapter_data[field]:
                logger.warning(f"Chapter missing required field: {field}")
                return False
        
        # Log missing optional fields
        for field in optional_but_useful:
            if field not in chapter_data:
                logger.info(f"Chapter missing optional field: {field}")
        
        return True
    
    def process_chapters_to_documents(self, chapters: List[Dict]) -> List[Document]:
        """
        Convert chapter data into LangChain Document objects.
        
        This method creates documents that preserve chapter context while
        making the content searchable. Each document contains both the
        full chapter content and rich metadata.
        
        Args:
            chapters: List of chapter data dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for chapter in chapters:
            # Extract chapter information
            chapter_num = chapter.get('number', 0)
            chapter_title = chapter.get('title', 'Unknown Chapter')
            content = chapter.get('content', '')
            
            # Create comprehensive metadata
            metadata = {
                'chapter_number': chapter_num,
                'chapter_title': chapter_title,
                'source_file': chapter.get('source_file', ''),
                'filename': chapter.get('filename', ''),
                'word_count': chapter.get('word_count', 0),
                'character_count': chapter.get('character_count', 0),
                'paragraph_count': chapter.get('paragraph_count', 0),
                'reading_time_minutes': chapter.get('reading_time_minutes', 0),
                'language': chapter.get('language', 'unknown'),
                'url': chapter.get('url', ''),
                'scraped_at': chapter.get('scraped_at', ''),
                'prev_chapter': chapter.get('prev_chapter'),
                'next_chapter': chapter.get('next_chapter'),
                'document_type': 'chapter',
                'book_series': self._extract_series_name(chapter.get('url', '')),
                'processed_at': datetime.now().isoformat()
            }
            
            # Create the main document with full chapter content
            # We prefix with chapter info for better context
            full_content = f"Chapter {chapter_num}: {chapter_title}\n\n{content}"
            
            document = Document(
                page_content=full_content,
                metadata=metadata
            )
            
            documents.append(document)
            
        logger.info(f"Converted {len(chapters)} chapters to {len(documents)} documents")
        return documents
    
    def _extract_series_name(self, url: str) -> str:
        """
        Extract series/book name from URL if possible.
        
        Args:
            url: URL string from chapter data
            
        Returns:
            Extracted series name or 'unknown'
        """
        if not url:
            return 'unknown'
        
        # Try to extract book name from URL pattern like '/book/series-name/chapter-X'
        match = re.search(r'/book/([^/]+)/', url)
        if match:
            series_name = match.group(1).replace('-', ' ').title()
            return series_name
        
        return 'unknown'
    
    def create_enhanced_chunks(self, documents: List[Document]) -> List[Dict]:
        """
        Create enhanced chunks with chapter-specific metadata.
        
        This method splits chapters into smaller, searchable chunks while
        preserving crucial context about which chapter and section each
        chunk comes from. This is essential for accurate source attribution.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of enhanced chunk dictionaries
        """
        enhanced_chunks = []
        
        for doc in documents:
            # Split the document into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            chapter_num = doc.metadata.get('chapter_number', 0)
            chapter_title = doc.metadata.get('chapter_title', 'Unknown')
            
            for i, chunk in enumerate(chunks):
                # Check dialogue once
                contains_dialogue = '"' in chunk.page_content or '"' in chunk.page_content
                
                # Create enhanced metadata for each chunk
                chunk_metadata = {
                    **doc.metadata,  # Inherit all chapter metadata
                    'chunk_index': i,
                    'total_chunks_in_chapter': len(chunks),
                    'chunk_id': f"ch{chapter_num:05d}_chunk{i:03d}",
                    'char_count': len(chunk.page_content),
                    'chunk_type': self._classify_chunk_content(chunk.page_content, contains_dialogue),
                    'contains_dialogue': contains_dialogue,
                    'contains_action': self._contains_action_words(chunk.page_content),
                    'is_chapter_start': i == 0,
                    'is_chapter_end': i == len(chunks) - 1
                }
                
                # Add character and location detection if possible
                chunk_metadata.update(self._extract_content_features(chunk.page_content))
                
                enhanced_chunk = {
                    'id': chunk_metadata['chunk_id'],
                    'text': chunk.page_content,
                    'metadata': chunk_metadata
                }
                
                enhanced_chunks.append(enhanced_chunk)
        
        logger.info(f"Created {len(enhanced_chunks)} enhanced chunks from {len(documents)} chapters")
        return enhanced_chunks
    
    def _classify_chunk_content(self, text: str, contains_dialogue: bool = None) -> str:
        """
        Classify the type of content in a chunk.
        
        Args:
            text: Chunk text content
            contains_dialogue: Pre-computed dialogue check to avoid redundant work
            
        Returns:
            Content type classification
        """
        text_lower = text.lower()
        
        # Use pre-computed dialogue check or compute once
        if contains_dialogue is None:
            contains_dialogue = '"' in text or '"' in text or "'" in text
            
        # Check for dialogue with more robust detection
        if contains_dialogue:
            # More sophisticated dialogue detection
            if (text_lower.count('"') >= 2 or text_lower.count('"') >= 2 or 
                '—' in text or '–' in text):  # em dash, en dash for dialogue
                return 'dialogue'
        
        # Check for action/description
        action_words = ['walked', 'ran', 'looked', 'turned', 'opened', 'closed', 'grabbed', 'touched']
        if any(word in text_lower for word in action_words):
            return 'action'
        
        # Check for internal monologue/thoughts
        if any(phrase in text_lower for phrase in ['thought', 'wondered', 'realized', 'remembered']):
            return 'thoughts'
        
        # Check for scene description
        descriptive_words = ['room', 'wall', 'door', 'window', 'light', 'shadow', 'color']
        if any(word in text_lower for word in descriptive_words):
            return 'description'
        
        return 'narrative'
    
    def _contains_action_words(self, text: str) -> bool:
        """
        Check if text contains action-indicating words.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if action words are present
        """
        action_indicators = [
            'walked', 'ran', 'moved', 'turned', 'looked', 'saw', 'heard',
            'opened', 'closed', 'grabbed', 'touched', 'felt', 'stood',
            'sat', 'fell', 'jumped', 'climbed', 'threw', 'caught'
        ]
        
        text_lower = text.lower()
        return any(action in text_lower for action in action_indicators)
    
    def _extract_content_features(self, text: str) -> Dict:
        """
        Extract additional content features from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted features
        """
        from collections import Counter
        
        features = {}
        
        # Efficient character name detection using Counter
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_counts = Counter(words)
        character_candidates = [
            word for word, count in word_counts.most_common() 
            if count > 1 and len(word) > 3
        ]
        features['potential_characters'] = character_candidates[:5]  # Limit to top 5
        
        # Emotion detection
        emotion_words = {
            'fear': ['afraid', 'scared', 'terrified', 'horror', 'panic'],
            'pain': ['hurt', 'pain', 'ache', 'agony', 'suffering'],
            'confusion': ['confused', 'puzzled', 'bewildered', 'lost', 'uncertain'],
            'surprise': ['surprised', 'shocked', 'amazed', 'stunned', 'astonished']
        }
        
        detected_emotions = []
        text_lower = text.lower()
        for emotion, keywords in emotion_words.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        features['emotions'] = detected_emotions
        
        # Time indicators
        time_indicators = ['morning', 'afternoon', 'evening', 'night', 'dawn', 'dusk', 'midnight']
        detected_times = [time for time in time_indicators if time in text_lower]
        features['time_context'] = detected_times
        
        return features
    
    def create_chapter_summary(self, chapter: Dict) -> str:
        """
        Create a brief summary of a chapter for quick reference.
        
        Args:
            chapter: Chapter data dictionary
            
        Returns:
            Chapter summary string
        """
        title = chapter.get('title', 'Unknown Chapter')
        number = chapter.get('number', '?')
        word_count = chapter.get('word_count', 0)
        
        # Get first few sentences of content for preview
        content = chapter.get('content', '')
        sentences = re.split(r'[.!?]+', content)
        preview = '. '.join(sentences[:2]).strip()
        if len(preview) > 200:
            preview = preview[:200] + "..."
        
        summary = f"Chapter {number}: {title}\n"
        summary += f"Length: {word_count} words\n"
        summary += f"Preview: {preview}"
        
        return summary


class ChapterRAGSystem:
    """
    Specialized RAG system for chapter-based content.
    
    This extends the base RAG system with chapter-specific functionality
    like cross-chapter search, character tracking, and plot analysis.
    """
    
    def __init__(self, base_rag_system, chapter_processor: ChapterProcessor):
        self.rag_system = base_rag_system
        self.chapter_processor = chapter_processor
        self.chapters_loaded = []
        self.answer_cache = {}  # Simple in-memory cache for answers
        self.cache_max_size = 1000  # Limit cache size
        
    def _create_cache_key(self, question: str, chapter_numbers: Optional[List[int]] = None) -> str:
        """
        Create a cache key for a question and chapter subset.
        """
        chapters_str = str(sorted(chapter_numbers)) if chapter_numbers else "all"
        combined = f"{question}|{chapters_str}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
        
    def _get_cached_answer(self, cache_key: str) -> Optional[Dict]:
        """
        Get cached answer if available and not expired.
        """
        if cache_key in self.answer_cache:
            cached_data = self.answer_cache[cache_key]
            # Cache for 1 hour
            if time.time() - cached_data['timestamp'] < 3600:
                logger.info(f"Using cached answer for key: {cache_key[:8]}...")
                return cached_data['data']
            else:
                # Remove expired entry
                del self.answer_cache[cache_key]
        return None
        
    def _cache_answer(self, cache_key: str, answer_data: Dict) -> None:
        """
        Cache an answer with timestamp.
        """
        # Simple LRU: remove oldest entries if cache is full
        if len(self.answer_cache) >= self.cache_max_size:
            oldest_key = min(self.answer_cache.keys(), 
                           key=lambda k: self.answer_cache[k]['timestamp'])
            del self.answer_cache[oldest_key]
            
        self.answer_cache[cache_key] = {
            'data': answer_data,
            'timestamp': time.time()
        }
        logger.info(f"Cached answer for key: {cache_key[:8]}...")
        
    def index_chapters(self, directory_path: str, file_pattern: str = "chapter-*.json", batch_size: int = 50) -> int:
        """
        Index all chapter files from a directory with batched embedding generation.
        
        Args:
            directory_path: Path to directory containing chapter files
            file_pattern: Glob pattern for chapter files
            batch_size: Size of batches for embedding generation to control memory usage
            
        Returns:
            Number of chunks created
        """
        try:
            # Load chapter files
            chapters = self.chapter_processor.load_chapter_files(directory_path, file_pattern)
            self.chapters_loaded = chapters
            
            # Convert to documents
            documents = self.chapter_processor.process_chapters_to_documents(chapters)
            
            # Create enhanced chunks
            chunks = self.chapter_processor.create_enhanced_chunks(documents)
            
            # Generate embeddings in batches to control memory usage
            texts = [chunk['text'] for chunk in chunks]
            total_chunks = len(chunks)
            
            logger.info(f"Generating embeddings for {total_chunks} chunks in batches of {batch_size}")
            
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                batch_texts = texts[i:batch_end]
                batch_chunks = chunks[i:batch_end]
                
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
                
                # Generate embeddings for this batch
                batch_embeddings = self.rag_system.embedding_service.embed_texts(batch_texts)
                
                # Store batch in vector database immediately
                self.rag_system.vector_store.add_documents(batch_chunks, batch_embeddings)
                
                # Optional: Small delay to prevent overwhelming the system
                import time
                time.sleep(0.1)
            
            logger.info(f"Successfully indexed {total_chunks} chunks from {len(chapters)} chapters")
            return total_chunks
            
        except Exception as e:
            logger.error(f"Error indexing chapters: {str(e)}")
            raise
    
    def query_with_chapter_context(self, question: str, 
                                  specific_chapters: Optional[List[int]] = None,
                                  use_reranking: bool = True) -> Dict:
        """
        Query the system with optional chapter filtering and caching.
        
        Args:
            question: Question to ask
            specific_chapters: Optional list of chapter numbers to search within
            use_reranking: Whether to use reranking
            
        Returns:
            Enhanced query results with chapter context
        """
        # Check cache first
        cache_key = self._create_cache_key(question, specific_chapters)
        cached_result = self._get_cached_answer(cache_key)
        if cached_result:
            return cached_result
            
        # Create filter if specific chapters requested
        filter_conditions = None
        if specific_chapters:
            from qdrant_client.http.models import Filter, FieldCondition, MatchAny
            
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="chapter_number",
                        match=MatchAny(values=specific_chapters)
                    )
                ]
            )
        
        # Perform the query with filtering
        result = self._query_with_filter(question, filter_conditions, use_reranking)
        
        # Enhance results with chapter context
        if result and result.get('sources'):
            result['chapter_summary'] = self._create_result_summary(result['sources'])
            result['chapters_referenced'] = list(set([
                source['metadata'].get('chapter_number') 
                for source in result['sources'] 
                if source['metadata'].get('chapter_number')
            ]))
        
        # Cache the result
        if result:
            self._cache_answer(cache_key, result)
        
        return result
    
    def _query_with_filter(self, question: str, filter_conditions, use_reranking: bool) -> Dict:
        """
        Internal method to perform filtered query with detailed timing.
        """
        try:
            start_time = time.time()
            
            # Embed the query
            embed_start = time.time()
            query_embedding = self.rag_system.embedding_service.embed_query(question)
            embed_time = time.time() - embed_start
            
            # Retrieve documents with filter
            retrieval_start = time.time()
            initial_k = self.rag_system.config.INITIAL_RETRIEVAL_K if use_reranking else self.rag_system.config.FINAL_RETRIEVAL_K
            search_results = self.rag_system.vector_store.search(
                query_embedding,
                limit=initial_k,
                filter_conditions=filter_conditions
            )
            retrieval_time = time.time() - retrieval_start
            
            if not search_results:
                logger.info(f"No results found - Embed: {embed_time:.3f}s, Retrieval: {retrieval_time:.3f}s")
                return {
                    'answer': "I couldn't find any relevant information in the specified chapters to answer your question.",
                    'sources': [],
                    'confidence': 0.0,
                    'processing_time': time.time() - start_time
                }
            
            # Rerank if requested
            rerank_time = 0
            if use_reranking:
                rerank_start = time.time()
                final_results = self.rag_system.reranker.rerank(
                    question,
                    search_results,
                    self.rag_system.config.FINAL_RETRIEVAL_K
                )
                rerank_time = time.time() - rerank_start
            else:
                final_results = search_results[:self.rag_system.config.FINAL_RETRIEVAL_K]
            
            # Generate answer
            llm_start = time.time()
            answer = self._generate_chapter_aware_answer(question, final_results)
            llm_time = time.time() - llm_start
            
            processing_time = time.time() - start_time
            avg_score = np.mean([doc['score'] for doc in final_results[:3]])
            confidence = min(avg_score * 100, 95)
            
            # Log detailed timing
            logger.info(
                f"Query processed - Total: {processing_time:.3f}s | "
                f"Embed: {embed_time:.3f}s | Retrieval: {retrieval_time:.3f}s | "
                f"Rerank: {rerank_time:.3f}s | LLM: {llm_time:.3f}s | "
                f"Results: {len(final_results)} | Confidence: {confidence:.1f}%"
            )
            
            return {
                'answer': answer,
                'sources': [
                    {
                        'text': doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'],
                        'score': doc['score'],
                        'chapter_number': doc.get('metadata', {}).get('chapter_number', 0),
                        'chapter_title': doc.get('metadata', {}).get('chapter_title', 'Unknown'),
                        'chunk_index': doc.get('metadata', {}).get('chunk_index', 0),
                        'chunk_type': doc.get('metadata', {}).get('chunk_type', 'narrative'),
                        'filename': doc.get('metadata', {}).get('filename', 'unknown'),
                        'metadata': doc.get('metadata', {})
                    }
                    for doc in final_results[:5]
                ],
                'confidence': confidence,
                'processing_time': processing_time,
                'timing_breakdown': {
                    'embed_time': embed_time,
                    'retrieval_time': retrieval_time,
                    'rerank_time': rerank_time,
                    'llm_time': llm_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing chapter query: {str(e)}")
            raise
    
    def _generate_chapter_aware_answer(self, question: str, relevant_docs: List[Dict]) -> str:
        """
        Generate answer with chapter-specific context awareness.
        """
        import tiktoken
        
        # Token limits
        MAX_TOTAL_PROMPT_TOKENS = 8000  # Conservative limit
        MAX_CHUNK_TOKENS = 300  # Limit per chunk
        MAX_CHUNKS = 8  # Limit number of chunks
        SYSTEM_PROMPT_TOKENS = 100  # Rough estimate for system prompt
        QUESTION_TOKENS = 50  # Rough estimate for question
        
        try:
            encoding = tiktoken.encoding_for_model(self.rag_system.config.OPENAI_MODEL)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")  # Fallback
        
        # Limit number of docs and deduplicate by chunk_id
        seen_chunks = set()
        unique_docs = []
        for doc in relevant_docs[:MAX_CHUNKS]:
            chunk_id = doc.get('metadata', {}).get('chunk_id', f"doc_{len(unique_docs)}")
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_docs.append(doc)
        
        # Group sources by chapter for better organization
        chapters_referenced = {}
        for doc in unique_docs:
            chapter_num = doc['metadata'].get('chapter_number', 0)
            chapter_title = doc['metadata'].get('chapter_title', 'Unknown')
            
            if chapter_num not in chapters_referenced:
                chapters_referenced[chapter_num] = {
                    'title': chapter_title,
                    'sources': []
                }
            
            chapters_referenced[chapter_num]['sources'].append(doc)
        
        # Create organized context with token limits
        context_parts = []
        total_context_tokens = 0
        available_tokens = MAX_TOTAL_PROMPT_TOKENS - SYSTEM_PROMPT_TOKENS - QUESTION_TOKENS
        
        for chapter_num in sorted(chapters_referenced.keys()):
            chapter_info = chapters_referenced[chapter_num]
            chapter_header = f"=== Chapter {chapter_num}: {chapter_info['title']} ==="
            
            # Check if we can fit the chapter header
            header_tokens = len(encoding.encode(chapter_header))
            if total_context_tokens + header_tokens > available_tokens:
                break
                
            context_parts.append(chapter_header)
            total_context_tokens += header_tokens
            
            for i, doc in enumerate(chapter_info['sources'], 1):
                chunk_type = doc['metadata'].get('chunk_type', 'narrative')
                
                # Truncate text if it's too long
                text = doc['text']
                text_tokens = len(encoding.encode(text))
                
                if text_tokens > MAX_CHUNK_TOKENS:
                    # Truncate to fit within token limit
                    tokens = encoding.encode(text)[:MAX_CHUNK_TOKENS]
                    text = encoding.decode(tokens) + "..."
                    text_tokens = MAX_CHUNK_TOKENS
                
                chunk_part = f"[{chunk_type.title()} - Part {i}]: {text}"
                chunk_tokens = len(encoding.encode(chunk_part))
                
                # Check if we can fit this chunk
                if total_context_tokens + chunk_tokens > available_tokens:
                    break
                    
                context_parts.append(chunk_part)
                total_context_tokens += chunk_tokens
            
            context_parts.append("")  # Add spacing between chapters
            total_context_tokens += 1  # For the newline
        
        context = "\n".join(context_parts)
        
        # Create chapter-aware prompt
        chapter_numbers = sorted(chapters_referenced.keys())
        chapter_range = f"Chapter {min(chapter_numbers)}" if len(chapter_numbers) == 1 else f"Chapters {min(chapter_numbers)}-{max(chapter_numbers)}"
        
        prompt = f"""Based on the following excerpts from {chapter_range} of the story, please answer the question. When referencing information, please mention which chapter it comes from.

Story Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the story excerpts above. Reference specific chapters when possible (e.g., "In Chapter 1..." or "As described in Chapter 3..."). If the context doesn't contain enough information to fully answer the question, please acknowledge this limitation."""
        
        # Log token usage for observability
        final_prompt_tokens = len(encoding.encode(prompt))
        logger.info(f"Prompt tokens: {final_prompt_tokens}, Context tokens: {total_context_tokens}, Chunks used: {len(unique_docs)}")
        
        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.rag_system.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant helping readers understand a story. Always cite specific chapters when referencing information and maintain narrative coherence in your explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=600,  # Use max_completion_tokens for newer models
                timeout=30,  # Added timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating chapter-aware answer: {str(e)}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."
    
    def _create_result_summary(self, sources: List[Dict]) -> str:
        """
        Create a summary of which chapters were referenced.
        """
        chapters = set()
        for source in sources:
            chapter_num = source.get('chapter_number', 0)
            chapter_title = source.get('chapter_title', 'Unknown')
            if chapter_num:
                chapters.add(f"Chapter {chapter_num}: {chapter_title}")
        
        if not chapters:
            return "No specific chapters identified."
        
        return f"Information drawn from: {', '.join(sorted(chapters))}"
    
    def get_chapter_list(self) -> List[Dict]:
        """
        Get a list of all loaded chapters with basic info.
        """
        return [
            {
                'number': ch.get('number', 0),
                'title': ch.get('title', 'Unknown'),
                'word_count': ch.get('word_count', 0),
                'filename': ch.get('filename', 'unknown')
            }
            for ch in self.chapters_loaded
        ]