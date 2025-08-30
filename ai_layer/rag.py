"""
Production-grade Retrieval-Augmented Generation (RAG) system module.

This module implements a complete RAG pipeline for answering questions about user files
stored in chunked form in Azure Cosmos DB with advanced features like token management,
conversation context, caching, confidence scoring, and streaming support.

Integrates with the search.py module for document retrieval.
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

# Import from search module
from .search import search_documents, search_by_file_id, SearchError
from utils.openai_client import get_embedding, answer_query_rag

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_TOP_K = 10
MAX_TOKEN_LIMIT = 8000  # Conservative limit for most models
CACHE_EXPIRY_HOURS = 24
CONFIDENCE_THRESHOLD = 0.7
SIMILARITY_WEIGHT = 0.6
CHUNK_DENSITY_WEIGHT = 0.4


@dataclass
class ConversationContext:
    """Conversation context for multi-turn RAG."""
    user_id: str
    conversation_id: str
    previous_queries: List[str]
    previous_answers: List[str]
    context_chunks: List[dict]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CacheEntry:
    """Cache entry for query-answer pairs."""
    query_hash: str
    answer: str
    chunks_used: int
    source_files: List[str]
    confidence_score: float
    timestamp: datetime
    user_id: str
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() - self.timestamp > timedelta(hours=CACHE_EXPIRY_HOURS)


@dataclass
class RagResult:
    """Enhanced RAG result with confidence and streaming support."""
    answer: str
    query: str
    chunks_used: int
    source_files: List[str]
    confidence_score: float
    context_used: bool
    cached: bool
    processing_time: float
    token_count: int


class RagError(Exception):
    """Custom exception for RAG-related errors."""
    
    def __init__(self, message: str, error_code: str = "RAG_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class TokenManager:
    """Manages token limits and chunk truncation."""
    
    def __init__(self, max_tokens: int = MAX_TOKEN_LIMIT):
        self.max_tokens = max_tokens
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def truncate_chunks_by_tokens(self, chunks: List[str], similarities: List[float]) -> Tuple[List[str], int]:
        """
        Truncate chunks based on token limits, prioritizing by similarity.
        
        Args:
            chunks: List of text chunks
            similarities: Corresponding similarity scores
            
        Returns:
            Tuple of (truncated_chunks, total_tokens)
        """
        logger.debug(f"Truncating {len(chunks)} chunks with token limit {self.max_tokens}")
        
        # Sort chunks by similarity (descending)
        chunk_similarity_pairs = list(zip(chunks, similarities))
        chunk_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected_chunks = []
        total_tokens = 0
        
        for chunk, similarity in chunk_similarity_pairs:
            chunk_tokens = self.estimate_tokens(chunk)
            
            if total_tokens + chunk_tokens <= self.max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                logger.debug(f"Added chunk (similarity: {similarity:.3f}, tokens: {chunk_tokens})")
            else:
                logger.debug(f"Skipped chunk (would exceed token limit)")
        
        logger.info(f"Selected {len(selected_chunks)} chunks totaling {total_tokens} tokens")
        return selected_chunks, total_tokens


class ConversationManager:
    """Manages conversation context for multi-turn RAG."""
    
    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}
        self.max_context_length = 5  # Keep last 5 exchanges
    
    def get_context_key(self, user_id: str, conversation_id: str) -> str:
        """Generate context key."""
        return f"{user_id}:{conversation_id}"
    
    def get_context(self, user_id: str, conversation_id: str) -> Optional[ConversationContext]:
        """Retrieve conversation context."""
        key = self.get_context_key(user_id, conversation_id)
        context = self.contexts.get(key)
        
        if context and datetime.utcnow() - context.timestamp > timedelta(hours=1):
            # Context expired
            del self.contexts[key]
            return None
            
        return context
    
    def update_context(self, user_id: str, conversation_id: str, query: str, 
                      answer: str, chunks: List[dict]) -> None:
        """Update conversation context."""
        key = self.get_context_key(user_id, conversation_id)
        
        if key in self.contexts:
            context = self.contexts[key]
            context.previous_queries.append(query)
            context.previous_answers.append(answer)
            context.context_chunks.extend(chunks)
            context.timestamp = datetime.utcnow()
            
            # Keep only recent exchanges
            if len(context.previous_queries) > self.max_context_length:
                context.previous_queries = context.previous_queries[-self.max_context_length:]
                context.previous_answers = context.previous_answers[-self.max_context_length:]
        else:
            context = ConversationContext(
                user_id=user_id,
                conversation_id=conversation_id,
                previous_queries=[query],
                previous_answers=[answer],
                context_chunks=chunks,
                timestamp=datetime.utcnow()
            )
            self.contexts[key] = context
    
    def build_context_prompt(self, context: ConversationContext) -> str:
        """Build context prompt from conversation history."""
        if not context.previous_queries:
            return ""
        
        context_parts = ["Previous conversation context:"]
        for i, (q, a) in enumerate(zip(context.previous_queries[:-1], context.previous_answers[:-1])):
            context_parts.append(f"Q{i+1}: {q}")
            context_parts.append(f"A{i+1}: {a[:200]}...")  # Truncate previous answers
        
        return "\n".join(context_parts) + "\n\nCurrent question:"


class CacheManager:
    """Manages caching for query-answer pairs."""
    
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
    
    def generate_cache_key(self, query: str, user_id: str, filters: Optional[Dict] = None) -> str:
        """Generate cache key for query."""
        cache_data = {
            "query": query.lower().strip(),
            "user_id": user_id,
            "filters": filters or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_answer(self, query: str, user_id: str, filters: Optional[Dict] = None) -> Optional[CacheEntry]:
        """Retrieve cached answer if available and not expired."""
        cache_key = self.generate_cache_key(query, user_id, filters)
        entry = self.cache.get(cache_key)
        
        if entry and not entry.is_expired():
            logger.info(f"Cache hit for query: {query[:50]}...")
            return entry
        elif entry:
            # Remove expired entry
            del self.cache[cache_key]
            
        return None
    
    def cache_answer(self, query: str, user_id: str, answer: str, chunks_used: int,
                    source_files: List[str], confidence_score: float, 
                    filters: Optional[Dict] = None) -> None:
        """Cache answer for future use."""
        cache_key = self.generate_cache_key(query, user_id, filters)
        
        entry = CacheEntry(
            query_hash=cache_key,
            answer=answer,
            chunks_used=chunks_used,
            source_files=source_files,
            confidence_score=confidence_score,
            timestamp=datetime.utcnow(),
            user_id=user_id
        )
        
        self.cache[cache_key] = entry
        logger.debug(f"Cached answer for query: {query[:50]}...")
    
    def clear_expired_cache(self) -> None:
        """Clear expired cache entries."""
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        for key in expired_keys:
            del self.cache[key]
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")


class ConfidenceCalculator:
    """Calculates confidence scores for RAG results."""
    
    def calculate_confidence(self, similarities: List[float], chunks: List[dict]) -> float:
        """
        Calculate confidence score based on similarity scores and chunk density.
        
        Args:
            similarities: List of cosine similarity scores
            chunks: List of retrieved chunks
            
        Returns:
            Confidence score between 0 and 1
        """
        if not similarities:
            return 0.0
        
        # Average similarity score
        avg_similarity = np.mean(similarities)
        
        # Chunk density score (how many high-quality chunks)
        high_quality_chunks = sum(1 for sim in similarities if sim > 0.7)
        chunk_density = high_quality_chunks / len(similarities)
        
        # Combined confidence score
        confidence = (SIMILARITY_WEIGHT * avg_similarity + 
                     CHUNK_DENSITY_WEIGHT * chunk_density)
        
        logger.debug(f"Confidence calculation: avg_sim={avg_similarity:.3f}, "
                    f"chunk_density={chunk_density:.3f}, final={confidence:.3f}")
        
        return min(confidence, 1.0)


# Global instances
token_manager = TokenManager()
conversation_manager = ConversationManager()
cache_manager = CacheManager()
confidence_calculator = ConfidenceCalculator()


def validate_intent(intent: Dict[str, Any]) -> None:
    """
    Validate the intent dictionary structure and required fields.
    
    Args:
        intent: Dictionary containing query intent and parameters
        
    Raises:
        RagError: If validation fails
    """
    logger.debug(f"Validating intent: {intent}")
    
    # Check if intent is a dictionary
    if not isinstance(intent, dict):
        raise RagError("Intent must be a dictionary", "INVALID_INTENT_TYPE")
    
    # Required fields
    required_fields = ["action", "query_text", "user_id"]
    for field in required_fields:
        if field not in intent:
            raise RagError(f"Missing required field: {field}", "MISSING_REQUIRED_FIELD")
        if not intent[field]:
            raise RagError(f"Required field '{field}' cannot be empty", "EMPTY_REQUIRED_FIELD")
    
    # Validate action
    if intent["action"] != "rag":
        raise RagError(f"Invalid action: {intent['action']}. Expected 'rag'", "INVALID_ACTION")
    
    # Validate query_text
    if not isinstance(intent["query_text"], str):
        raise RagError("query_text must be a string", "INVALID_QUERY_TEXT_TYPE")
    
    # Validate user_id
    if not isinstance(intent["user_id"], str):
        raise RagError("user_id must be a string", "INVALID_USER_ID_TYPE")
    
    # Validate optional fields if present
    if "top_k" in intent and intent["top_k"] is not None:
        if not isinstance(intent["top_k"], int) or intent["top_k"] <= 0:
            raise RagError("top_k must be a positive integer", "INVALID_TOP_K")
    
    if "file_id" in intent and intent["file_id"] is not None:
        if not isinstance(intent["file_id"], str):
            raise RagError("file_id must be a string", "INVALID_FILE_ID_TYPE")
    
    if "platform" in intent and intent["platform"] is not None:
        if not isinstance(intent["platform"], str):
            raise RagError("platform must be a string", "INVALID_PLATFORM_TYPE")
    
    logger.info("Intent validation successful")


def build_search_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build search intent dictionary compatible with search.py from RAG intent.
    
    Args:
        intent: RAG intent dictionary
        
    Returns:
        Search intent dictionary compatible with search_documents function
    """
    logger.debug("Building search intent from RAG intent")
    
    search_intent = {
        "query_text": intent["query_text"],
        "user_id": intent["user_id"]
    }
    
    # Add optional fields that match search.py expectations
    if intent.get("platform"):
        search_intent["platform"] = intent["platform"]
    
    if intent.get("file_type"):
        search_intent["file_type"] = intent["file_type"]
    
    if intent.get("time_range"):
        search_intent["time_range"] = intent["time_range"]
    
    # Handle offset and limit for pagination
    if intent.get("offset"):
        search_intent["offset"] = intent["offset"]
    
    if intent.get("limit"):
        search_intent["limit"] = intent["limit"]
    
    logger.debug(f"Built search intent: {search_intent}")
    return search_intent


def extract_text_and_similarities(chunks: List[dict]) -> Tuple[List[str], List[float]]:
    """
    Extract text content and similarity scores from chunk documents returned by search.py.
    
    Args:
        chunks: List of enhanced chunk documents from search_documents function
        
    Returns:
        Tuple of (text_chunks, similarity_scores)
        
    Raises:
        RagError: If no valid text content is found
    """
    logger.debug(f"Extracting text and similarities from {len(chunks)} chunks")
    
    text_chunks = []
    similarities = []
    
    # Fields expected from search.py enhanced results
    text_fields = ["text", "content", "chunk_text", "body", "text_content"]
    
    for i, chunk in enumerate(chunks):
        # Extract text content
        chunk_text = None
        for field in text_fields:
            if field in chunk and chunk[field]:
                chunk_text = str(chunk[field]).strip()
                break
        
        # Extract similarity score from search.py enhanced results
        similarity = chunk.get('_similarity', 0.0)  # search.py adds _similarity field
        
        if chunk_text:
            text_chunks.append(chunk_text)
            similarities.append(similarity)
            logger.debug(f"Extracted chunk {i}: {len(chunk_text)} chars, similarity: {similarity:.3f}")
        else:
            logger.warning(f"No valid text content found in chunk {i}")
    
    if not text_chunks:
        raise RagError("No valid text content found in any chunks", "NO_TEXT_CONTENT")
    
    logger.info(f"Successfully extracted {len(text_chunks)} chunks with similarities")
    return text_chunks, similarities


def extract_source_files(chunks: List[dict]) -> List[str]:
    """
    Extract and deduplicate source file names from chunk metadata.
    
    Args:
        chunks: List of enhanced chunk documents from search.py
        
    Returns:
        List of unique source file names
    """
    logger.debug(f"Extracting source files from {len(chunks)} chunks")
    
    source_files = set()
    filename_fields = ["fileName", "filename", "title", "source_file", "file_name", "document_name"]
    
    for chunk in chunks:
        for field in filename_fields:
            if field in chunk and chunk[field]:
                filename = str(chunk[field]).strip()
                if filename:
                    source_files.add(filename)
                    break
    
    result = list(source_files)
    logger.info(f"Found {len(result)} unique source files: {result}")
    return result


def answer_query_with_rag(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to answer a query using RAG with advanced features.
    
    Args:
        intent: Dictionary containing:
            - action (str): Must be "rag"
            - query_text (str): The question to answer
            - user_id (str): User identifier
            - conversation_id (str, optional): For multi-turn conversations
            - file_id (str, optional): Filter by specific file
            - platform (str, optional): Filter by platform
            - file_type (str, optional): Filter by file type
            - time_range (str|dict, optional): Filter by time range
            - top_k (int, optional): Number of chunks to retrieve (default: 10)
            - use_cache (bool, optional): Enable caching (default: True)
            - stream (bool, optional): Enable streaming (default: False) # Changed default
            - include_confidence (bool, optional): Include confidence score (default: True)
    
    Returns:
        Dictionary containing:
            - answer (str): Generated answer
            - query (str): Original query text
            - chunks_used (int): Number of chunks used
            - source_files (list): List of source file names
            - confidence_score (float): Confidence score (0-1)
            - context_used (bool): Whether conversation context was used
            - cached (bool): Whether result was cached
            - processing_time (float): Processing time in seconds
            - token_count (int): Total tokens used
    
    Raises:
        RagError: If any step in the RAG process fails
    """
    start_time = time.time()
    logger.info(f"Starting enhanced RAG query for user: {intent.get('user_id')}")
    
    try:
        # Step 1: Validate intent
        validate_intent(intent)
        
        query_text = intent["query_text"]
        user_id = intent["user_id"]
        conversation_id = intent.get("conversation_id")
        top_k = intent.get("top_k", DEFAULT_TOP_K)
        use_cache = intent.get("use_cache", True)
        stream = intent.get("stream", False)  # Changed default to False
        include_confidence = intent.get("include_confidence", True)
        
        logger.info(f"Processing query: '{query_text[:100]}...' for user: {user_id}")
        
        # Step 2: Check cache first
        search_intent = build_search_intent(intent)
        cached_result = None
        
        if use_cache:
            cached_result = cache_manager.get_cached_answer(query_text, user_id, search_intent)
            if cached_result:
                processing_time = time.time() - start_time
                return {
                    "answer": cached_result.answer,
                    "query": query_text,
                    "chunks_used": cached_result.chunks_used,
                    "source_files": cached_result.source_files,
                    "confidence_score": cached_result.confidence_score,
                    "context_used": False,
                    "cached": True,
                    "processing_time": processing_time,
                    "token_count": token_manager.estimate_tokens(cached_result.answer)
                }
        
        # Step 3: Get conversation context
        context = None
        context_used = False
        if conversation_id:
            context = conversation_manager.get_context(user_id, conversation_id)
            context_used = context is not None
        
        # Step 4: Enhance query with context if available
        enhanced_query = query_text
        if context:
            context_prompt = conversation_manager.build_context_prompt(context)
            enhanced_query = f"{context_prompt}\n{query_text}"
            # Update search intent with enhanced query
            search_intent["query_text"] = enhanced_query
        
        # Step 5: Retrieve relevant chunks using search.py
        logger.debug(f"Searching for similar documents with top_k={top_k}")
        try:
            if intent.get("file_id"):
                # Search within a specific file
                chunks = search_by_file_id(
                    file_id=intent["file_id"],
                    user_id=user_id,
                    query_text=enhanced_query,
                    top_k=top_k
                )
            else:
                # General document search
                chunks = search_documents(search_intent, top_k=top_k)
            
            logger.info(f"Retrieved {len(chunks)} chunks from search module")
            
        except SearchError as e:
            logger.error(f"Search error: {e}")
            raise RagError(f"Failed to retrieve documents: {str(e)}", "SEARCH_ERROR")
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            raise RagError(f"Unexpected error during search: {str(e)}", "SEARCH_ERROR")
        
        # Step 6: Check if chunks were found
        if not chunks:
            logger.warning("No relevant chunks found for the query")
            raise RagError("No relevant documents found for your query", "NO_CHUNKS_FOUND")
        
        # Step 7: Extract text and similarities
        text_chunks, similarities = extract_text_and_similarities(chunks)
        
        # Step 8: Apply token limits with smart ranking
        final_chunks, token_count = token_manager.truncate_chunks_by_tokens(text_chunks, similarities)
        final_similarities = similarities[:len(final_chunks)]
        
        # Step 9: Calculate confidence score
        confidence_score = 0.0
        if include_confidence:
            confidence_score = confidence_calculator.calculate_confidence(final_similarities, chunks[:len(final_chunks)])
        
        # Step 10: Extract source files
        source_files = extract_source_files(chunks[:len(final_chunks)])
        
        # Step 11: Generate RAG answer
        logger.debug("Generating RAG answer using OpenAI client")
        try:
            # FIXED: Handle streaming vs non-streaming properly
            if stream:
                # For streaming, we need to handle differently
                answer_generator = answer_query_rag(
                    chunks=final_chunks,
                    query=query_text,
                    stream=True
                )
                # Collect all chunks into a single answer
                answer_parts = []
                for chunk in answer_generator:
                    if isinstance(chunk, str):
                        answer_parts.append(chunk)
                    elif isinstance(chunk, dict) and 'content' in chunk:
                        answer_parts.append(chunk['content'])
                answer = ''.join(answer_parts)
            else:
                # Non-streaming mode
                answer = answer_query_rag(
                    chunks=final_chunks,
                    query=query_text,
                    stream=False
                )
            
            # Ensure answer is a string
            if not isinstance(answer, str):
                raise RagError("Generated answer is not a string", "INVALID_ANSWER_TYPE")
            
            logger.info("RAG answer generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise RagError(f"Failed to generate answer: {str(e)}", "ANSWER_GENERATION_ERROR")
        
        # Step 12: Update conversation context
        if conversation_id:
            conversation_manager.update_context(user_id, conversation_id, query_text, answer, chunks[:len(final_chunks)])
        
        # Step 13: Cache the result (only if confidence is high enough)
        if use_cache and confidence_score > CONFIDENCE_THRESHOLD:
            cache_manager.cache_answer(
                query_text, user_id, answer, len(final_chunks), 
                source_files, confidence_score, search_intent
            )
        
        # Step 14: Prepare response
        processing_time = time.time() - start_time
        
        # FIXED: Calculate token count safely
        answer_token_count = token_manager.estimate_tokens(answer) if isinstance(answer, str) else 0
        
        response = {
            "answer": answer,
            "query": query_text,
            "chunks_used": len(final_chunks),
            "source_files": source_files,
            "confidence_score": confidence_score,
            "context_used": context_used,
            "cached": False,
            "processing_time": processing_time,
            "token_count": token_count + answer_token_count  # Total tokens used
        }
        
        logger.info(f"Enhanced RAG process completed successfully. "
                   f"Used {len(final_chunks)} chunks, confidence: {confidence_score:.3f}, "
                   f"processing time: {processing_time:.2f}s")
        return response
        
    except RagError:
        # Re-raise RagError as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in RagError
        logger.error(f"Unexpected error in RAG process: {e}")
        raise RagError(f"Unexpected error in RAG process: {str(e)}", "UNEXPECTED_ERROR")


def stream_rag_response(intent: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Stream RAG response for real-time processing.
    
    Args:
        intent: Same as answer_query_with_rag but forces streaming
        
    Yields:
        Dictionary chunks containing partial responses
    """
    logger.info("Starting streaming RAG response")
    
    try:
        # Force streaming mode for this function
        streaming_intent = intent.copy()
        streaming_intent["stream"] = True
        
        # Follow similar steps as main function but handle streaming
        validate_intent(streaming_intent)
        
        query_text = streaming_intent["query_text"]
        user_id = streaming_intent["user_id"]
        top_k = streaming_intent.get("top_k", DEFAULT_TOP_K)
        
        # Quick cache check
        search_intent = build_search_intent(streaming_intent)
        cached_result = cache_manager.get_cached_answer(query_text, user_id, search_intent)
        
        if cached_result:
            # Stream cached result
            yield {
                "type": "chunk",
                "content": cached_result.answer,
                "metadata": {
                    "chunks_used": cached_result.chunks_used,
                    "confidence_score": cached_result.confidence_score,
                    "cached": True
                }
            }
            yield {
                "type": "complete",
                "result": {
                    "answer": cached_result.answer,
                    "cached": True,
                    "chunks_used": cached_result.chunks_used,
                    "confidence_score": cached_result.confidence_score
                }
            }
            return
        
        # Proceed with normal RAG process for streaming
        # Get chunks
        if streaming_intent.get("file_id"):
            chunks = search_by_file_id(
                file_id=streaming_intent["file_id"],
                user_id=user_id,
                query_text=query_text,
                top_k=top_k
            )
        else:
            chunks = search_documents(search_intent, top_k=top_k)
        
        if not chunks:
            yield {
                "type": "error",
                "error": "No relevant documents found"
            }
            return
        
        # Process chunks
        text_chunks, similarities = extract_text_and_similarities(chunks)
        final_chunks, token_count = token_manager.truncate_chunks_by_tokens(text_chunks, similarities)
        
        # Stream the answer generation
        answer_generator = answer_query_rag(
            chunks=final_chunks,
            query=query_text,
            stream=True
        )
        
        full_answer = ""
        for chunk in answer_generator:
            if isinstance(chunk, str):
                content = chunk
            elif isinstance(chunk, dict) and 'content' in chunk:
                content = chunk['content']
            else:
                continue
                
            full_answer += content
            yield {
                "type": "chunk",
                "content": content,
                "metadata": {
                    "chunks_used": len(final_chunks),
                    "token_count": token_count
                }
            }
        
        # Final result
        confidence_score = confidence_calculator.calculate_confidence(similarities[:len(final_chunks)], chunks[:len(final_chunks)])
        source_files = extract_source_files(chunks[:len(final_chunks)])
        
        yield {
            "type": "complete",
            "result": {
                "answer": full_answer,
                "query": query_text,
                "chunks_used": len(final_chunks),
                "source_files": source_files,
                "confidence_score": confidence_score,
                "cached": False,
                "token_count": token_count
            }
        }
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield {
            "type": "error",
            "error": str(e)
        }

def clear_user_cache(user_id: str) -> int:
    """Clear cache for a specific user."""
    cleared = 0
    keys_to_remove = [k for k, v in cache_manager.cache.items() if v.user_id == user_id]
    
    for key in keys_to_remove:
        del cache_manager.cache[key]
        cleared += 1
    
    logger.info(f"Cleared {cleared} cache entries for user: {user_id}")
    return cleared


def get_system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    return {
        "cache_size": len(cache_manager.cache),
        "active_conversations": len(conversation_manager.contexts),
        "token_limit": token_manager.max_tokens,
        "cache_expiry_hours": CACHE_EXPIRY_HOURS,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


def health_check() -> Dict[str, Any]:
    """
    Perform a health check on the enhanced RAG system.
    
    Returns:
        Dictionary containing health status information
    """
    logger.info("Performing enhanced RAG system health check")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "module": "enhanced_rag",
        "version": "2.0.0",
        "features": {
            "token_management": True,
            "conversation_context": True,
            "caching": True,
            "confidence_scoring": True,
            "streaming": True
        }
    }
    
    try:
        # Check if required modules are available
        from utils.openai_client import get_embedding, answer_query_rag
        from .search import search_documents, search_by_file_id
        
        health_status["dependencies"] = {
            "embedding": "available",
            "search": "available",
            "answer_generation": "available"
        }
        
        # Add system stats
        health_status["system_stats"] = get_system_stats()
        
        # Clear expired cache
        cache_manager.clear_expired_cache()
        
    except ImportError as e:
        logger.error(f"Dependency check failed: {e}")
        health_status["status"] = "unhealthy"
        health_status["error"] = f"Missing dependency: {str(e)}"
    
    logger.info(f"Health check completed: {health_status['status']}")
    return health_status


# Export main functions for external use
__all__ = [
    'answer_query_with_rag',
    'stream_rag_response',
    'clear_user_cache',
    'get_system_stats',
    'health_check',
    'RagError',
    'RagResult'
]


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    # Example intent with advanced features
    sample_intent = {
        "action": "rag",
        "query_text": "What is the main topic of the document?",
        "user_id": "user123",
        "conversation_id": "conv456",
        "platform": "google_drive",
        "file_type": "pdf",
        "time_range": "last_3_months",
        "top_k": 5,
        "use_cache": True,
        "stream": True,
        "include_confidence": True
    }
    
    try:
        result = answer_query_with_rag(sample_intent)
        print("Enhanced RAG Result:", result)
        
        # Test streaming
        print("\nStreaming response:")
        for chunk in stream_rag_response(sample_intent):
            print(f"Stream chunk: {chunk}")
            
    except RagError as e:
        print(f"RAG Error: {e.message} (Code: {e.error_code})")
    except Exception as e:
        print(f"Unexpected Error: {e}")