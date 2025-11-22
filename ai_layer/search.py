"""
search.py - Production-grade intelligent document search module for AI assistant.

This module provides hybrid vector + metadata search capabilities over document chunks
stored in Azure Cosmos DB. It combines semantic similarity search with metadata filtering
to deliver precise and contextually relevant document retrieval.

Key Features:
- Structured intent-based search with comprehensive validation
- Hybrid vector + metadata search over Azure Cosmos DB
- Enhanced result processing with metadata enrichment
- Production-ready error handling and logging
- Modular design with clear separation of concerns

Author: AI Assistant
Version: 1.0.0
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import os

# Import external dependencies
from utils.openai_client import get_embedding
from utils.cosmos_client import CosmosVectorClient

# Configure module logger
logger = logging.getLogger(__name__)

def create_cosmos_client(cosmos_endpoint: str = os.getenv('COSMOS_ENDPOINT'), cosmos_key: str = os.getenv('COSMOS_KEY')) -> CosmosVectorClient:
    """
    Factory function to create a CosmosVectorClient instance
    
    Args:
        cosmos_endpoint: Cosmos DB endpoint URL
        cosmos_key: Cosmos DB primary key
        
    Returns:
        Configured CosmosVectorClient instance
    """
    return CosmosVectorClient(cosmos_endpoint, cosmos_key)

class SearchError(Exception):
    """
    Custom exception for search-related errors.
    
    Raised for all predictable search issues including validation failures,
    embedding generation errors, and search execution problems.
    """
    pass

class Platform(Enum):
    """Supported platform types for document storage."""
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    DROPBOX = "dropbox"
    SHAREPOINT = "sharepoint"
    LOCAL = "local"
    SLACK = "slack"
    TEAMS = "teams"

class FileCategory(Enum):
    """Document file categories for enhanced metadata."""
    TEXT = "text"
    PDF = "pdf"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    OTHER = "other"

@dataclass
class SearchMetrics:
    """Container for search performance metrics."""
    total_results: int
    search_time_ms: float
    embedding_time_ms: float
    filter_count: int
    top_similarity_score: float

class SearchValidator:
    """
    Validates and normalizes search intent parameters.
    
    Ensures all required fields are present and valid, normalizes optional fields,
    and provides helpful error messages for validation failures.
    """
    
    # Supported MIME types for document search (matching cosmos_client expectations)
    SUPPORTED_MIME_TYPES = {
        'text/plain',
        'text/markdown',
        'text/html',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # .pptx
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
        'application/msword',  # .doc
        'application/vnd.ms-powerpoint',  # .ppt
        'application/vnd.ms-excel',  # .xls
        'application/json',
        'application/xml',
        'text/csv'
    }
    
    # File type shortcuts that will be converted to MIME types
    FILE_TYPE_SHORTCUTS = {
        'pdf': 'PDF',
        'doc': 'DOC',
        'docx': 'DOCX',
        'xls': 'XLS',
        'xlsx': 'XLSX',
        'ppt': 'PPT',
        'pptx': 'PPTX',
        'txt': 'TXT'
    }
    
    # Supported relative time ranges
    SUPPORTED_TIME_RANGES = {
        'last_hour', 'last_24_hours', 'last_7_days', 'last_30_days', 
        'last_month', 'last_3_months', 'last_6_months', 'last_year'
    }
    
    @classmethod
    def validate_intent(cls, intent: dict) -> dict:
        """
        Validates and normalizes the search intent dictionary.
        
        Args:
            intent: Raw search intent dictionary from client
            
        Returns:
            Normalized and validated intent dictionary
            
        Raises:
            SearchError: If validation fails for any required or optional field
        """
        if not isinstance(intent, dict):
            raise SearchError("Intent must be a dictionary")
        
        # Validate required fields
        cls._validate_required_fields(intent)
        
        # Create normalized copy
        normalized = intent.copy()
        
        # Normalize and validate optional fields
        cls._normalize_query_text(normalized)
        cls._validate_platform(normalized)
        cls._validate_file_type(normalized)
        cls._validate_time_range(normalized)
        cls._validate_pagination(normalized)
        
        logger.debug(f"Intent validation successful for user: {normalized['user_id']}")
        return normalized
    
    @classmethod
    def _validate_required_fields(cls, intent: dict) -> None:
        """Validates required fields in intent dictionary."""
        if 'query_text' not in intent:
            raise SearchError("Missing required field: query_text")
        
        if not intent['query_text'] or not isinstance(intent['query_text'], str):
            raise SearchError("query_text must be a non-empty string")
        
        if 'user_id' not in intent:
            raise SearchError("Missing required field: user_id")
        
        if not intent['user_id'] or not isinstance(intent['user_id'], str):
            raise SearchError("user_id must be a non-empty string")
    
    @classmethod
    def _normalize_query_text(cls, intent: dict) -> None:
        """Normalizes query text by trimming whitespace and validating length."""
        query_text = intent['query_text'].strip()
        
        if len(query_text) < 2:
            raise SearchError("query_text must be at least 2 characters long")
        
        if len(query_text) > 1000:
            raise SearchError("query_text cannot exceed 1000 characters")
        
        intent['query_text'] = query_text
    
    @classmethod
    def _validate_platform(cls, intent: dict) -> None:
        """Validates and normalizes platform field."""
        if 'platform' not in intent or not intent['platform']:
            return
        
        platform = intent['platform'].lower().strip()
        valid_platforms = {p.value for p in Platform}
        
        if platform not in valid_platforms:
            logger.warning(f"Unknown platform: {platform}. Supported: {valid_platforms}")
            # Don't raise error, just log warning to allow flexibility
        
        intent['platform'] = platform
    
    @classmethod
    def _validate_file_type(cls, intent: dict) -> None:
        """Validates file type field and converts to cosmos_client expected format."""
        if 'file_type' not in intent or not intent['file_type']:
            return
        
        file_type = intent['file_type'].lower().strip()
        
        # Convert shortcut to cosmos_client format
        if file_type in cls.FILE_TYPE_SHORTCUTS:
            intent['file_type'] = cls.FILE_TYPE_SHORTCUTS[file_type]
        elif file_type.upper() in ['PDF', 'DOC', 'DOCX', 'XLS', 'XLSX', 'PPT', 'PPTX', 'TXT']:
            intent['file_type'] = file_type.upper()
        elif file_type in cls.SUPPORTED_MIME_TYPES:
            # If it's already a MIME type, convert it to cosmos_client format
            mime_to_type = {
                'application/pdf': 'PDF',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
                'application/msword': 'DOC',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLSX',
                'application/vnd.ms-excel': 'XLS',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PPTX',
                'application/vnd.ms-powerpoint': 'PPT',
                'text/plain': 'TXT'
            }
            intent['file_type'] = mime_to_type.get(file_type, file_type.upper())
        else:
            logger.warning(f"Unknown file type: {file_type}. Will be passed as-is.")
            intent['file_type'] = file_type.upper()
    
    @classmethod
    def _validate_time_range(cls, intent: dict) -> None:
        """Validates time range specification."""
        if 'time_range' not in intent or not intent['time_range']:
            return
        
        time_range = intent['time_range']
        
        if isinstance(time_range, str):
            cls._validate_relative_time_range(time_range)
        elif isinstance(time_range, dict):
            cls._validate_absolute_time_range(time_range)
        else:
            raise SearchError("time_range must be a string or dictionary")
    
    @classmethod
    def _validate_relative_time_range(cls, time_range: str) -> None:
        """Validates relative time range strings."""
        if time_range not in cls.SUPPORTED_TIME_RANGES:
            raise SearchError(f"Invalid time range: {time_range}. Supported: {cls.SUPPORTED_TIME_RANGES}")
    
    @classmethod
    def _validate_absolute_time_range(cls, time_range: dict) -> None:
        """Validates absolute time range dictionaries."""
        if 'start_date' not in time_range and 'end_date' not in time_range:
            raise SearchError("Absolute time range must contain at least start_date or end_date")
        
        for date_field in ['start_date', 'end_date']:
            if date_field in time_range:
                try:
                    datetime.fromisoformat(time_range[date_field])
                except ValueError:
                    raise SearchError(f"Invalid {date_field} format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    
    @classmethod
    def _validate_pagination(cls, intent: dict) -> None:
        """Validates pagination parameters."""
        if 'offset' in intent:
            if not isinstance(intent['offset'], int) or intent['offset'] < 0:
                raise SearchError("offset must be a non-negative integer")
        
        if 'limit' in intent:
            if not isinstance(intent['limit'], int) or intent['limit'] < 1 or intent['limit'] > 100:
                raise SearchError("limit must be an integer between 1 and 100")

class FilterBuilder:
    """
    Constructs metadata filters for Cosmos DB search from validated intent.
    
    Translates high-level search intent into specific database query filters
    compatible with cosmos_client.py filter structure.
    """
    
    @classmethod
    def build_filters(cls, intent: dict) -> dict:
        """
        Builds comprehensive metadata filters compatible with CosmosVectorClient.
        
        Args:
            intent: Validated search intent dictionary
            
        Returns:
            Dictionary of filters compatible with cosmos_client.py structure:
            - user_id: string (required)
            - platforms: list of strings (optional)
            - file_types: list of strings (optional, in cosmos_client format)
            - time_range: ISO8601 string (optional)
        """
        filters = {}
        
        # Always include user_id for data isolation and security (required by cosmos_client)
        filters['user_id'] = intent['user_id']
        
        # Platform-specific filtering - convert to list format expected by cosmos_client
        if intent.get('platform'):
            filters['platforms'] = [intent['platform']]
        elif intent.get('platforms'):  # Support both single platform and list
            filters['platforms'] = intent['platforms'] if isinstance(intent['platforms'], list) else [intent['platforms']]
        
        # File type filtering - convert to list format expected by cosmos_client
        if intent.get('file_type'):
            filters['file_types'] = [intent['file_type']]
        elif intent.get('file_types'):  # Support both single file_type and list
            filters['file_types'] = intent['file_types'] if isinstance(intent['file_types'], list) else [intent['file_types']]
        
        # Time-based filtering - convert to ISO8601 string format
        if intent.get('time_range'):
            time_filter = cls._build_time_filter(intent['time_range'])
            if time_filter:
                filters['time_range'] = time_filter
        
        # Note: Removed filters not supported by cosmos_client:
        # - file_id (not in cosmos_client filter structure)
        # - file_name (not in cosmos_client filter structure)  
        # - content_type (not in cosmos_client filter structure)
        # - min_chunk_size (not in cosmos_client filter structure)
        
        logger.debug(f"Built {len(filters)} filters for cosmos_client: {list(filters.keys())}")
        return filters
    
    @classmethod
    def _build_time_filter(cls, time_range: Union[str, dict]) -> Optional[str]:
        """
        Builds time-based filter in ISO8601 string format for cosmos_client.
        
        Args:
            time_range: Time range specification (relative string or absolute dict)
            
        Returns:
            ISO8601 time string for created_at >= filtering or None if invalid
        """
        if isinstance(time_range, str):
            return cls._build_relative_time_filter(time_range)
        elif isinstance(time_range, dict):
            return cls._build_absolute_time_filter(time_range)
        
        return None
    
    @classmethod
    def _build_relative_time_filter(cls, time_range: str) -> Optional[str]:
        """Builds ISO8601 string for relative time ranges (e.g., 'last_7_days')."""
        now = datetime.utcnow()
        
        time_deltas = {
            'last_hour': timedelta(hours=1),
            'last_24_hours': timedelta(hours=24),
            'last_7_days': timedelta(days=7),
            'last_30_days': timedelta(days=30),
            'last_month': timedelta(days=30),
            'last_3_months': timedelta(days=90),
            'last_6_months': timedelta(days=180),
            'last_year': timedelta(days=365)
        }
        
        if time_range in time_deltas:
            start_time = now - time_deltas[time_range]
            return start_time.isoformat() + 'Z'
        
        logger.warning(f"Unknown relative time range: {time_range}")
        return None
    
    @classmethod
    def _build_absolute_time_filter(cls, time_range: dict) -> Optional[str]:
        """Builds ISO8601 string for absolute time ranges with start/end dates."""
        # cosmos_client only supports single time_range value, so use start_date if available
        if 'start_date' in time_range:
            try:
                start_date = datetime.fromisoformat(time_range['start_date'])
                return start_date.isoformat() + 'Z'
            except ValueError:
                logger.error(f"Invalid start_date format: {time_range['start_date']}")
        
        # If no start_date but has end_date, this is not supported by cosmos_client
        # Log warning and return None
        if 'end_date' in time_range:
            logger.warning("cosmos_client only supports start_date filtering (>=), end_date filtering not supported")
        
        return None

class SearchResultProcessor:
    """
    Processes and enhances raw search results from Cosmos DB.
    
    Adds metadata enrichment, content previews, similarity scoring,
    and result categorization to provide comprehensive search results.
    """
    
    # MIME type to file category mapping
    MIME_TO_CATEGORY = {
        'text/plain': FileCategory.TEXT,
        'text/markdown': FileCategory.TEXT,
        'text/html': FileCategory.TEXT,
        'application/pdf': FileCategory.PDF,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileCategory.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': FileCategory.PRESENTATION,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileCategory.SPREADSHEET,
        'application/msword': FileCategory.DOCUMENT,
        'application/vnd.ms-powerpoint': FileCategory.PRESENTATION,
        'application/vnd.ms-excel': FileCategory.SPREADSHEET,
        'image/': FileCategory.IMAGE,
        'audio/': FileCategory.AUDIO,
        'video/': FileCategory.VIDEO,
        'application/zip': FileCategory.ARCHIVE,
        'application/x-rar': FileCategory.ARCHIVE,
    }
    
    @classmethod
    def process_results(cls, results: List[dict], intent: dict) -> List[dict]:
        """
        Processes and enhances raw search results from cosmos_client.
        
        Args:
            results: Raw search results from CosmosVectorClient (with 'score' field)
            intent: Original search intent for context
            
        Returns:
            List of enhanced and sorted search results
        """
        if not results:
            logger.info("No search results to process")
            return []
        
        # Results from cosmos_client are already sorted by similarity score
        # Convert 'score' field to '_similarity' for consistency
        for result in results:
            if 'score' in result:
                # Convert vector distance to similarity (lower distance = higher similarity)
                # Assuming cosine distance (0-2 range), convert to similarity (0-1 range)
                distance = result.get('score', 1.0)
                similarity = max(0.0, 1.0 - (distance / 2.0))
                result['_similarity'] = similarity
            else:
                result['_similarity'] = 0.0
        
        # Enhance each result with additional metadata
        enhanced_results = []
        for idx, result in enumerate(results):
            try:
                enhanced_result = cls._enhance_result(result, intent, idx)
                enhanced_results.append(enhanced_result)
            except Exception as e:
                logger.error(f"Error enhancing result {idx}: {str(e)}")
                # Include original result if enhancement fails
                enhanced_results.append(result)
        
        logger.info(f"Processed {len(enhanced_results)} search results")
        return enhanced_results
    
    @classmethod
    def _enhance_result(cls, result: dict, intent: dict, rank: int) -> dict:
        """
        Enhances a single search result with additional metadata.
        
        Args:
            result: Single search result from CosmosVectorClient
            intent: Original search intent
            rank: Result ranking (0-based)
            
        Returns:
            Enhanced result with additional metadata
        """
        enhanced = result.copy()
        
        # Add search metadata
        enhanced['_search_metadata'] = {
            'query_text': intent['query_text'],
            'similarity_score': result.get('_similarity', 0.0),
            'rank': rank,
            'timestamp': datetime.utcnow().isoformat(),
            'matched_filters': cls._extract_matched_filters(result, intent)
        }
        
        # Add content preview
        enhanced['_preview'] = cls._generate_preview(result)
        
        # Add file category
        enhanced['_file_category'] = cls._determine_file_category(result)
        
        # Add relevance indicators
        enhanced['_relevance'] = cls._calculate_relevance_indicators(result, intent)
        
        # Add chunk context
        enhanced['_chunk_context'] = cls._build_chunk_context(result)
        
        return enhanced
    
    @classmethod
    def _extract_matched_filters(cls, result: dict, intent: dict) -> dict:
        """Extracts which filters were matched for this result."""
        matched = {}
        
        if intent.get('platform') and result.get('platform') == intent['platform']:
            matched['platform'] = result['platform']
        
        # Check file_type match using cosmos_client format
        if intent.get('file_type'):
            # cosmos_client returns mime_type, so we need to map back to check
            result_mime = result.get('mime_type', '')
            if cls._file_type_matches_mime(intent['file_type'], result_mime):
                matched['mime_type'] = result_mime
        
        return matched
    
    @classmethod
    def _file_type_matches_mime(cls, file_type: str, mime_type: str) -> bool:
        """Check if a file_type matches a mime_type."""
        # cosmos_client file type mappings (from cosmos_client.py)
        file_type_mappings = {
            'PDF': 'application/pdf',
            'DOCX': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'DOC': 'application/msword',
            'XLSX': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'XLS': 'application/vnd.ms-excel',
            'PPTX': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'PPT': 'application/vnd.ms-powerpoint',
            'TXT': 'text/plain'
        }
        
        expected_mime = file_type_mappings.get(file_type.upper())
        return expected_mime == mime_type
    
    @classmethod
    def _generate_preview(cls, result: dict, max_length: int = 200) -> str:
        """
        Generates a content preview snippet.
        
        Args:
            result: Search result containing text content
            max_length: Maximum preview length
            
        Returns:
            Preview snippet with ellipsis if truncated
        """
        # Try different content fields
        content_fields = ['text', 'content', 'summary', 'description']
        
        for field in content_fields:
            if field in result and result[field]:
                content = str(result[field]).strip()
                if content:
                    if len(content) <= max_length:
                        return content
                    else:
                        return content[:max_length] + '...'
        
        return "No preview available"
    
    @classmethod
    def _determine_file_category(cls, result: dict) -> str:
        """
        Determines the file category based on MIME type.
        
        Args:
            result: Search result containing mime_type
            
        Returns:
            File category string
        """
        mime_type = result.get('mime_type', '').lower()
        
        # Direct mapping
        if mime_type in cls.MIME_TO_CATEGORY:
            return cls.MIME_TO_CATEGORY[mime_type].value
        
        # Prefix matching for broad categories
        for prefix, category in cls.MIME_TO_CATEGORY.items():
            if prefix.endswith('/') and mime_type.startswith(prefix):
                return category.value
        
        # Fallback based on content patterns
        if 'word' in mime_type or 'document' in mime_type:
            return FileCategory.DOCUMENT.value
        elif 'presentation' in mime_type or 'powerpoint' in mime_type:
            return FileCategory.PRESENTATION.value
        elif 'spreadsheet' in mime_type or 'excel' in mime_type:
            return FileCategory.SPREADSHEET.value
        
        return FileCategory.OTHER.value
    
    @classmethod
    def _calculate_relevance_indicators(cls, result: dict, intent: dict) -> dict:
        """
        Calculates various relevance indicators for the result.
        
        Args:
            result: Search result
            intent: Original search intent
            
        Returns:
            Dictionary of relevance indicators
        """
        indicators = {
            'similarity_score': result.get('_similarity', 0.0),
            'is_recent': cls._is_recent_document(result),
            'has_keywords': cls._contains_keywords(result, intent['query_text']),
            'confidence_level': cls._calculate_confidence_level(result)
        }
        
        return indicators
    
    @classmethod
    def _is_recent_document(cls, result: dict, days_threshold: int = 30) -> bool:
        """Checks if document is recent based on created_at timestamp."""
        if 'created_at' not in result:
            return False
        
        try:
            created_at_str = result['created_at']
            # Handle both with and without 'Z' suffix
            if created_at_str.endswith('Z'):
                created_at_str = created_at_str[:-1] + '+00:00'
            
            created_at = datetime.fromisoformat(created_at_str)
            threshold = datetime.utcnow() - timedelta(days=days_threshold)
            return created_at.replace(tzinfo=None) > threshold
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def _contains_keywords(cls, result: dict, query_text: str) -> bool:
        """Checks if result contains keywords from the query."""
        content_fields = ['text', 'content', 'fileName', 'summary']
        query_keywords = query_text.lower().split()
        
        for field in content_fields:
            if field in result and result[field]:
                content = str(result[field]).lower()
                for keyword in query_keywords:
                    if keyword in content:
                        return True
        
        return False
    
    @classmethod
    def _calculate_confidence_level(cls, result: dict) -> str:
        """Calculates confidence level based on similarity score."""
        similarity = result.get('_similarity', 0.0)
        
        if similarity >= 0.8:
            return 'high'
        elif similarity >= 0.6:
            return 'medium'
        elif similarity >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    @classmethod
    def _build_chunk_context(cls, result: dict) -> dict:
        """Builds context information about the chunk."""
        return {
            'chunk_index': result.get('chunk_index', 0),
            'total_chunks': result.get('total_chunks', 1),
            'chunk_size': len(str(result.get('text', result.get('content', '')))),
            'has_next_chunk': result.get('chunk_index', 0) < result.get('total_chunks', 1) - 1,
            'has_previous_chunk': result.get('chunk_index', 0) > 0
        }
    
client = create_cosmos_client()

def search_documents(intent: dict, top_k: int = 10) -> List[dict]:
    """
    Main entry point for intelligent document search.
    
    Performs hybrid vector + metadata search over document chunks stored in Azure Cosmos DB.
    Combines semantic similarity search with metadata filtering to deliver precise and
    contextually relevant document retrieval.
    
    Args:
        intent: Search intent dictionary containing:
            - query_text (str, required): The search query text
            - user_id (str, required): User id
            - platform (str, optional): Platform filter (google_drive, onedrive, etc.)
            - file_type (str, optional): File type filter (PDF, DOCX, etc.)
            - time_range (str|dict, optional): Time range filter
        top_k: Maximum number of results to return (default: 10, max: 100)
    
    Returns:
        List of enhanced document chunks sorted by similarity score, containing:
        - Original chunk data (text/content, metadata, etc.)
        - _similarity: Similarity score from vector search
        - _search_metadata: Search context and matched filters
        - _preview: Content snippet preview
        - _file_category: Categorized file type
        - _relevance: Relevance indicators
        - _chunk_context: Chunk positioning information
    
    Raises:
        SearchError: For all predictable search issues including:
            - Input validation failures
            - Embedding generation errors
            - Search execution problems
    
    Example:
        >>> intent = {
        ...     'query_text': 'quarterly sales report analysis',
        ...     'user_id': 'user123',
        ...     'platform': 'google_drive',
        ...     'file_type': 'pdf',
        ...     'time_range': 'last_3_months'
        ... }
        >>> results = search_documents(intent, top_k=5)
        >>> for result in results:
        ...     print(f"File: {result['fileName']}")
        ...     print(f"Score: {result['_similarity']:.3f}")
        ...     print(f"Preview: {result['_preview'][:100]}...")
    """
    start_time = datetime.utcnow()
    client = create_cosmos_client()
    
    # Validate top_k parameter
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        raise SearchError("top_k must be an integer between 1 and 100")
    
    logger.info(f"Starting document search for user: {intent.get('user_id', 'unknown')}")
    
    try:
        # Step 1: Validate and normalize search intent
        logger.debug("Validating search intent")
        validated_intent = SearchValidator.validate_intent(intent)
        logger.debug(f"Validated search intent: {validated_intent}")

        # Step 2: Generate embedding for query text
        logger.debug(f"Generating embedding for query: '{validated_intent['query_text']}'")
        embedding_start = datetime.utcnow()
        
        query_embedding = get_embedding(validated_intent['query_text'])
        
        embedding_time = (datetime.utcnow() - embedding_start).total_seconds() * 1000
        
        if not query_embedding or not isinstance(query_embedding, list):
            raise SearchError("Failed to generate valid embedding for query text")
        
        logger.debug(f"Generated embedding vector of length: {len(query_embedding)}")
        
        # Step 3: Build metadata filters compatible with cosmos_client
        logger.debug("Building metadata filters for cosmos_client")
        filters = FilterBuilder.build_filters(validated_intent)
        
        # Step 4: Perform vector search using cosmos_client
        logger.debug(f"Executing vector search with {len(filters)} filters and top_k={top_k}")
        search_start = datetime.utcnow()
        
        raw_results = client.vector_search_cosmos(
            embedding=query_embedding,
            filters=filters,
            top_k=top_k
        )

        search_time = (datetime.utcnow() - search_start).total_seconds() * 1000
        
        if not isinstance(raw_results, list):
            raise SearchError("Invalid response format from vector search")
        
        logger.info(f"Vector search returned {len(raw_results)} results")
        
        # Step 5: Process and enhance search results
        logger.debug("Processing and enhancing search results")
        enhanced_results = SearchResultProcessor.process_results(raw_results, validated_intent)
        
        # Step 6: Calculate search metrics
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        metrics = SearchMetrics(
            total_results=len(enhanced_results),
            search_time_ms=search_time,
            embedding_time_ms=embedding_time,
            filter_count=len(filters),
            top_similarity_score=enhanced_results[0]['_similarity'] if enhanced_results else 0.0
        )
        
        logger.info(f"Search completed successfully in {total_time:.2f}ms: "
                   f"{metrics.total_results} results, "
                   f"top score: {metrics.top_similarity_score:.3f}")
        
        # Add metrics to first result for debugging (optional)
        if enhanced_results:
            enhanced_results[0]['_search_metrics'] = {
                'total_time_ms': total_time,
                'search_time_ms': metrics.search_time_ms,
                'embedding_time_ms': metrics.embedding_time_ms,
                'filter_count': metrics.filter_count
            }
        
        return enhanced_results
        
    except SearchError:
        # Re-raise SearchError without modification
        raise
    except Exception as e:
        error_msg = f"Unexpected error during document search: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise SearchError(error_msg) from e


def search_by_file_id(file_id: str, user_id: str, query_text: str = "", top_k: int = 10) -> List[dict]:
    """
    Search within a specific document by file ID.

    Performs vector search limited to chunks from a specific document,
    useful for exploring content within a known file.

    Args:
        file_id: Unique identifier for the file
        user_id: User identifier for data isolation
        query_text: Optional query text for semantic search within the file
        top_k: Maximum number of results to return

    Returns:
        List of enhanced document chunks from the specified file

    Raises:
        SearchError: If file_id is invalid or search fails
    """
    if not file_id or not isinstance(file_id, str):
        raise SearchError("file_id must be a non-empty string")

    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")

    logger.info(f"Searching within file {file_id} for user {user_id}")

    try:
        # CRITICAL FIX: Use Cosmos DB direct query instead of vector search filtering
        # The vector search filtering might not be working correctly for file_id

        cosmos_client = create_cosmos_client()

        # First, try to find any chunks from this file using direct Cosmos query
        query = """
            SELECT * FROM c 
            WHERE c.user_id = @user_id 
            AND (c.file_id = @file_id OR c.fileId = @file_id)
            AND IS_DEFINED(c.text)
        """

        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@file_id", "value": file_id}
        ]

        # Execute direct query to find file chunks
        direct_results = list(cosmos_client.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
            max_item_count=top_k * 2  # Get more than needed for filtering
        ))

        logger.info(f"Direct query found {len(direct_results)} chunks for file_id: {file_id}")

        if not direct_results:
            # Try alternative file_id patterns that might be stored
            alt_patterns = [
                file_id.replace('.pdf', ''),  # Remove extension
                file_id.replace('_removed', ''),  # Remove _removed suffix
                file_id.replace(' ', '_'),  # Space to underscore
                file_id.replace('_', ' '),  # Underscore to space
            ]

            for alt_file_id in alt_patterns:
                if alt_file_id != file_id:
                    logger.info(f"Trying alternative file_id pattern: {alt_file_id}")
                    alt_params = [
                        {"name": "@user_id", "value": user_id},
                        {"name": "@file_id", "value": alt_file_id}
                    ]

                    alt_results = list(cosmos_client.container.query_items(
                        query=query,
                        parameters=alt_params,
                        enable_cross_partition_query=True,
                        max_item_count=top_k * 2
                    ))

                    if alt_results:
                        logger.info(f"Found {len(alt_results)} chunks with alternative file_id: {alt_file_id}")
                        direct_results = alt_results
                        break

        if not direct_results:
            logger.warning(f"No chunks found for any file_id patterns for: {file_id}")
            return []

        # If we have a query_text, perform semantic search on the found chunks
        if query_text.strip():
            # Generate embedding for the query
            query_embedding = get_embedding(query_text)

            # Use vector search with the found chunks as a post-filter
            raw_results = client.vector_search_cosmos(
                embedding=query_embedding,
                filters={'user_id': user_id},
                top_k=top_k * 3  # Get more results to filter
            )

            # Filter vector results to only include chunks from our target file
            file_results = []
            target_chunk_ids = {result.get('id') for result in direct_results}

            for result in raw_results:
                if result.get('id') in target_chunk_ids:
                    file_results.append(result)
                    if len(file_results) >= top_k:
                        break

            if not file_results:
                # Fallback: return direct results if vector search filtering fails
                file_results = direct_results[:top_k]
                # Add dummy similarity scores
                for result in file_results:
                    result['_similarity'] = 0.5
        else:
            # No query text - return all chunks from the file
            file_results = direct_results[:top_k]
            # Add dummy similarity scores for consistency
            for result in file_results:
                result['_similarity'] = 1.0

        # Process results using the existing result processor
        intent = {
            'query_text': query_text or "file content",
            'user_id': user_id,
            'file_id': file_id
        }

        enhanced_results = SearchResultProcessor.process_results(file_results, intent)

        logger.info(f"Found {len(enhanced_results)} chunks in file {file_id}")
        return enhanced_results

    except Exception as e:
        error_msg = f"Error searching file {file_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise SearchError(error_msg) from e


def get_similar_documents(file_id: str, user_id: str, top_k: int = 5) -> List[dict]:
    """
    Find documents similar to a given file.
    
    Uses the average embedding of chunks from the source file to find
    semantically similar documents in the user's collection.
    
    Args:
        file_id: Source file identifier
        user_id: User identifier for data isolation
        top_k: Maximum number of similar documents to return
    
    Returns:
        List of similar documents with similarity scores
    
    Raises:
        SearchError: If source file not found or search fails
    """
    if not file_id or not isinstance(file_id, str):
        raise SearchError("file_id must be a non-empty string")
    
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    logger.info(f"Finding documents similar to {file_id} for user {user_id}")
    
    try:
        # First, get chunks from the source file to create average embedding
        source_chunks = search_by_file_id(file_id, user_id, query_text="", top_k=50)
        
        if not source_chunks:
            raise SearchError(f"Source file {file_id} not found or has no content")
        
        # Extract embeddings from source chunks (if available)
        # Note: This assumes cosmos_client returns embeddings, which it might not
        # Alternative: Use a representative text sample for embedding
        
        # Get representative text from source file
        representative_text = " ".join([
            chunk.get('text', chunk.get('content', ''))[:200] 
            for chunk in source_chunks[:3]  # Use first 3 chunks
        ])
        
        if not representative_text.strip():
            raise SearchError(f"Unable to extract representative text from file {file_id}")
        
        # Generate embedding for representative text
        source_embedding = get_embedding(representative_text)
        
        # Search for similar documents
        filters = {'user_id': user_id}
        
        raw_results = client.vector_search_cosmos(
            embedding=source_embedding,
            filters=filters,
            top_k=top_k * 10  # Get more to filter out source file
        )
        
        # Filter out chunks from the source file and group by file
        file_groups = {}
        for result in raw_results:
            result_file_id = result.get('file_id') or result.get('fileId')
            
            # Skip chunks from the source file
            if result_file_id == file_id:
                continue
                
            if result_file_id not in file_groups:
                file_groups[result_file_id] = {
                    'file_id': result_file_id,
                    'fileName': result.get('fileName', 'Unknown'),
                    'platform': result.get('platform', 'unknown'),
                    'mime_type': result.get('mime_type', ''),
                    'chunks': [],
                    'max_similarity': 0.0
                }
            
            # Convert score to similarity
            distance = result.get('score', 1.0)
            similarity = max(0.0, 1.0 - (distance / 2.0))
            result['_similarity'] = similarity
            
            file_groups[result_file_id]['chunks'].append(result)
            file_groups[result_file_id]['max_similarity'] = max(
                file_groups[result_file_id]['max_similarity'], 
                similarity
            )
        
        # Convert to list and sort by max similarity
        similar_files = list(file_groups.values())
        similar_files.sort(key=lambda x: x['max_similarity'], reverse=True)
        
        # Limit results and add metadata
        similar_files = similar_files[:top_k]
        
        for file_info in similar_files:
            file_info['_similarity'] = file_info['max_similarity']
            file_info['chunk_count'] = len(file_info['chunks'])
            file_info['_file_category'] = SearchResultProcessor._determine_file_category(file_info)
        
        logger.info(f"Found {len(similar_files)} similar documents to {file_id}")
        return similar_files
        
    except SearchError:
        raise
    except Exception as e:
        error_msg = f"Error finding similar documents to {file_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise SearchError(error_msg) from e


def get_search_suggestions(partial_query: str, user_id: str, limit: int = 5) -> List[str]:
    """
    Generate search suggestions based on partial query input.
    
    Uses existing document content to suggest relevant search terms
    and completions for the user's partial input.
    
    Args:
        partial_query: Partial search query text
        user_id: User identifier for personalized suggestions
        limit: Maximum number of suggestions to return
    
    Returns:
        List of suggested search query completions
    
    Raises:
        SearchError: If suggestion generation fails
    """
    if not partial_query or len(partial_query.strip()) < 2:
        return []
    
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    logger.debug(f"Generating search suggestions for: '{partial_query}'")
    
    try:
        # Generate embedding for partial query
        query_embedding = get_embedding(partial_query)
        
        # Search for relevant documents
        filters = {'user_id': user_id}
        
        results = client.vector_search_cosmos(
            embedding=query_embedding,
            filters=filters,
            top_k=20  # Get more results for suggestion generation
        )
        
        # Extract keywords and phrases from relevant documents
        suggestions = set()
        query_words = set(partial_query.lower().split())
        
        for result in results:
            # Extract text content
            text_content = result.get('text', result.get('content', ''))
            file_name = result.get('fileName', '')
            
            # Add filename-based suggestions
            if file_name and partial_query.lower() in file_name.lower():
                suggestions.add(file_name.replace('.', ' ').replace('_', ' '))
            
            # Extract phrases from content
            if text_content:
                # Simple phrase extraction (can be enhanced with NLP)
                sentences = text_content.split('.')[:3]  # First 3 sentences
                for sentence in sentences:
                    words = sentence.strip().split()
                    if len(words) >= 3 and len(words) <= 8:
                        phrase = ' '.join(words)
                        if any(word in phrase.lower() for word in query_words):
                            suggestions.add(phrase.strip())
        
        # Convert to list and sort by relevance (simple length-based for now)
        suggestion_list = list(suggestions)
        suggestion_list.sort(key=lambda x: (len(x.split()), x))
        
        # Filter and limit results
        filtered_suggestions = [
            s for s in suggestion_list 
            if len(s) <= 100 and len(s.split()) <= 10
        ]
        
        return filtered_suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Error generating search suggestions: {str(e)}")
        return []  # Return empty list instead of raising error for suggestions


def get_search_stats(user_id: str) -> dict:
    """
    Get search and document statistics for a user.
    
    Provides insights into the user's document collection and search patterns.
    
    Args:
        user_id: User identifier
    
    Returns:
        Dictionary containing user's search statistics
    
    Raises:
        SearchError: If statistics retrieval fails
    """
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    logger.info(f"Retrieving search statistics for user: {user_id}")
    
    try:
        # Get a sample of user's documents to analyze
        sample_results = client.vector_search_cosmos(
            embedding=[0.0] * 1536,  # Zero vector for broad sampling
            filters={'user_id': user_id},
            top_k=100  # Sample size
        )
        
        if not sample_results:
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'platforms': {},
                'file_types': {},
                'recent_documents': 0,
                'average_chunk_size': 0
            }
        
        # Analyze the sample
        platforms = {}
        file_types = {}
        total_chunk_size = 0
        recent_count = 0
        unique_files = set()
        
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        for result in sample_results:
            # Track unique files
            file_id = result.get('file_id') or result.get('fileId')
            if file_id:
                unique_files.add(file_id)
            
            # Platform statistics
            platform = result.get('platform', 'unknown')
            platforms[platform] = platforms.get(platform, 0) + 1
            
            # File type statistics
            mime_type = result.get('mime_type', 'unknown')
            file_types[mime_type] = file_types.get(mime_type, 0) + 1
            
            # Chunk size analysis
            text_content = result.get('text', result.get('content', ''))
            total_chunk_size += len(text_content)
            
            # Recent documents
            if 'created_at' in result:
                try:
                    created_at_str = result['created_at']
                    if created_at_str.endswith('Z'):
                        created_at_str = created_at_str[:-1] + '+00:00'
                    
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at.replace(tzinfo=None) > thirty_days_ago:
                        recent_count += 1
                except (ValueError, TypeError):
                    pass
        
        # Calculate statistics
        total_chunks = len(sample_results)
        avg_chunk_size = total_chunk_size // total_chunks if total_chunks > 0 else 0
        
        stats = {
            'total_documents': len(unique_files),
            'total_chunks': total_chunks,
            'platforms': platforms,
            'file_types': file_types,
            'recent_documents': recent_count,
            'average_chunk_size': avg_chunk_size,
            'sample_size': total_chunks,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Generated statistics for user {user_id}: {stats['total_documents']} documents")
        return stats
        
    except Exception as e:
        error_msg = f"Error retrieving search statistics: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise SearchError(error_msg) from e


def search_files_by_name(file_name: str, user_id: str, platform: Optional[str] = None,
                         exact_match: bool = False, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for files by name using exact or partial matching.

    This function leverages the enhanced cosmos_client functionality for better
    performance and relevance scoring.

    Args:
        file_name: Name of the file to search for (can be partial)
        user_id: User ID to search within
        platform: Optional platform filter
        exact_match: Whether to perform exact match (True) or partial match (False)
        limit: Maximum number of results to return

    Returns:
        List of matching file metadata with file_id, fileName, platform, and other details

    Raises:
        SearchError: If search fails
    """
    if not file_name or not isinstance(file_name, str):
        raise SearchError("file_name must be a non-empty string")

    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")

    logger.info(f"Searching for files by name: '{file_name}' for user {user_id}")

    try:
        # Create cosmos client
        cosmos_client = create_cosmos_client()

        # Build filters for the enhanced cosmos_client function
        filters = {}
        if platform:
            filters['platforms'] = [platform]

        # Use the enhanced cosmos_client search_files_by_name function
        results = cosmos_client.search_files_by_name(
            file_name_query=file_name,
            user_id=user_id,
            exact_match=exact_match,
            filters=filters if filters else None
        )

        # Limit results if needed (cosmos_client already limits to 50)
        if len(results) > limit:
            results = results[:limit]

        # Enhance results for search.py compatibility
        enhanced_results = []
        for result in results:
            enhanced_result = {
                'file_id': result.get('file_id'),
                'fileName': result.get('fileName'),
                'platform': result.get('platform'),
                'mime_type': result.get('mime_type'),
                'created_at': result.get('created_at'),
                'metadata': result.get('metadata', {}),
                'file_category': result.get('file_type', 'Unknown'),
                'chunk_count': result.get('chunk_count', 0),
                'search_relevance': result.get('search_relevance', 1.0)
            }
            enhanced_results.append(enhanced_result)

        logger.info(f"Found {len(enhanced_results)} files matching name '{file_name}'")
        return enhanced_results

    except Exception as e:
        error_msg = f"Error searching files by name '{file_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise SearchError(error_msg) from e


def search_files_by_name_advanced(file_name_query: str, user_id: str,
                                  exact_match: bool = False,
                                  include_fuzzy: bool = True,
                                  filters: Optional[Dict[str, Any]] = None,
                                  limit: int = 10) -> List[Dict[str, Any]]:
    """
    Advanced file name search combining exact, partial, and fuzzy matching.

    This function provides the most comprehensive file name search by combining
    multiple search strategies and returning ranked results.

    Args:
        file_name_query: File name or partial name to search for
        user_id: User ID to search within
        exact_match: If True, prioritize exact matches
        include_fuzzy: If True, include fuzzy matching for better recall
        filters: Additional filter criteria (platforms, file_types, time_range)
        limit: Maximum number of results to return

    Returns:
        List of matching files ranked by relevance with enhanced metadata

    Raises:
        SearchError: If search fails
    """
    if not file_name_query or not isinstance(file_name_query, str):
        raise SearchError("file_name_query must be a non-empty string")

    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")

    logger.info(f"Advanced file name search: '{file_name_query}' for user {user_id}")

    try:
        cosmos_client = create_cosmos_client()
        all_results = {}  # Use dict to avoid duplicates by file_id

        # Strategy 1: Exact match (if requested or as baseline)
        if exact_match:
            exact_results = cosmos_client.search_files_by_name(
                file_name_query=file_name_query,
                user_id=user_id,
                exact_match=True,
                filters=filters
            )
            for result in exact_results:
                file_id = result.get('file_id')
                if file_id:
                    result['_search_strategy'] = 'exact_match'
                    result['_boost_score'] = 1.0
                    all_results[file_id] = result

        # Strategy 2: Partial match
        partial_results = cosmos_client.search_files_by_name(
            file_name_query=file_name_query,
            user_id=user_id,
            exact_match=False,
            filters=filters
        )
        for result in partial_results:
            file_id = result.get('file_id')
            if file_id and file_id not in all_results:
                result['_search_strategy'] = 'partial_match'
                result['_boost_score'] = 0.8
                all_results[file_id] = result

        # Strategy 3: Fuzzy matching (if enabled)
        if include_fuzzy:
            fuzzy_results = cosmos_client.search_files_by_name_fuzzy(
                file_name_query=file_name_query,
                user_id=user_id,
                max_results=limit * 2,
                min_relevance=0.2  # Lower threshold for more results
            )
            for result in fuzzy_results:
                file_id = result.get('file_id')
                if file_id and file_id not in all_results:
                    result['_search_strategy'] = 'fuzzy_match'
                    result['_boost_score'] = 0.6
                    all_results[file_id] = result

        # Convert back to list and calculate final scores
        final_results = list(all_results.values())

        # Calculate combined relevance score
        for result in final_results:
            base_relevance = result.get('search_relevance', 0.5)
            boost_score = result.get('_boost_score', 0.5)

            # Combine base relevance with strategy boost
            final_score = (base_relevance * 0.7) + (boost_score * 0.3)
            result['_final_relevance'] = final_score

            # Add search metadata
            result['_advanced_search_metadata'] = {
                'strategy': result.get('_search_strategy', 'unknown'),
                'base_relevance': base_relevance,
                'boost_score': boost_score,
                'final_score': final_score
            }

        # Sort by final relevance score and limit results
        final_results.sort(key=lambda x: x.get('_final_relevance', 0), reverse=True)
        final_results = final_results[:limit]

        logger.info(f"Advanced search found {len(final_results)} files for '{file_name_query}'")
        return final_results

    except Exception as e:
        error_msg = f"Error in advanced file name search '{file_name_query}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise SearchError(error_msg) from e


def get_file_id_by_name(file_name: str, user_id: str, platform: Optional[str] = None) -> Optional[str]:
    """
    Get a file ID by its name for a specific user.

    This is a convenience function that uses search_files_by_name but returns
    only the file ID of the first matching file.

    Args:
        file_name: Name of the file to search for
        user_id: User ID to search within
        platform: Optional platform filter

    Returns:
        File ID if found, None otherwise
    """
    try:
        results = search_files_by_name(file_name, user_id, platform, exact_match=True, limit=1)
        if results:
            return results[0]['file_id']

        # If no exact match, try partial match
        results = search_files_by_name(file_name, user_id, platform, exact_match=False, limit=1)
        if results:
            return results[0]['file_id']

        return None
    except SearchError:
        return None
    except Exception as e:
        logger.error(f"Error getting file ID by name: {str(e)}")
        return None


def get_file_id_by_name_enhanced(file_name: str, user_id: str,
                                 platform: Optional[str] = None,
                                 confidence_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """
    Enhanced version of get_file_id_by_name that returns match confidence and metadata.

    Args:
        file_name: Name of the file to search for
        user_id: User ID to search within
        platform: Optional platform filter
        confidence_threshold: Minimum confidence score required for a match

    Returns:
        Dictionary with file_id, confidence score, and metadata if found, None otherwise
    """
    try:
        # Build filters
        filters = {}
        if platform:
            filters['platforms'] = [platform]

        # Try advanced search
        results = search_files_by_name_advanced(
            file_name_query=file_name,
            user_id=user_id,
            exact_match=False,
            include_fuzzy=True,
            filters=filters if filters else None,
            limit=1
        )

        if results:
            result = results[0]
            confidence = result.get('_final_relevance', 0.0)

            if confidence >= confidence_threshold:
                return {
                    'file_id': result.get('file_id'),
                    'fileName': result.get('fileName'),
                    'confidence': confidence,
                    'strategy': result.get('_search_strategy', 'unknown'),
                    'platform': result.get('platform'),
                    'metadata': result.get('metadata', {})
                }

        return None

    except Exception as e:
        logger.error(f"Error getting file ID by name: {str(e)}")
        return None

# Utility functions for external integrations

def validate_search_intent(intent: dict) -> tuple[bool, str]:
    """
    Validate search intent without raising exceptions.
    
    Args:
        intent: Search intent dictionary to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        SearchValidator.validate_intent(intent)
        return True, ""
    except SearchError as e:
        return False, str(e)


def create_search_intent(query_text: str, user_id: str, **kwargs) -> dict:
    """
    Helper function to create a properly formatted search intent.
    
    Args:
        query_text: The search query
        user_id: User identifier
        **kwargs: Additional optional parameters (platform, file_type, time_range, etc.)
    
    Returns:
        Properly formatted search intent dictionary
    """
    intent = {
        'query_text': query_text,
        'user_id': user_id
    }
    
    # Add optional parameters
    optional_fields = ['platform', 'file_type', 'time_range', 'offset', 'limit']
    for field in optional_fields:
        if field in kwargs and kwargs[field] is not None:
            intent[field] = kwargs[field]
    
    return intent


# Export main functions for external use
__all__ = [
    'search_documents',
    'search_by_file_id', 
    'get_similar_documents',
    'get_search_suggestions',
    'get_search_stats',
    'validate_search_intent',
    'create_search_intent',
    'SearchError',
    'Platform',
    'FileCategory',
    'search_files_by_name',  # Add this
    'get_file_id_by_name'
]


# Example usage and testing functions (remove in production)
if __name__ == "__main__":
    # Example usage
    test_intent = {
        'query_text': 'quarterly sales report',
        'user_id': 'test_user_123',
        'platform': 'google_drive',
        'file_type': 'pdf',
        'time_range': 'last_3_months'
    }
    
    try:
        print("Testing document search...")
        results = search_documents(test_intent, top_k=5)
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results[:3]):
            print(f"\nResult {i+1}:")
            print(f"  File: {result.get('fileName', 'Unknown')}")
            print(f"  Similarity: {result.get('_similarity', 0):.3f}")
            print(f"  Preview: {result.get('_preview', 'No preview')[:100]}...")
            
    except SearchError as e:
        print(f"Search error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")