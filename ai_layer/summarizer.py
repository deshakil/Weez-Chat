"""
Production-grade summarization module for Weez MCP AI Agent.

This module provides intelligent document summarization capabilities for files
stored across cloud platforms like Google Drive, OneDrive, Slack, and local storage.
Supports both file-based and query-based summarization with configurable detail levels.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from .search import search_documents, search_by_file_id, create_search_intent, SearchError
from utils.openai_client import summarize_chunks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Custom exception for summarization-related errors."""
    pass


def validate_intent(intent: Dict[str, Any]) -> None:
    """
    Validate the intent dictionary for summarization requirements.
    
    Args:
        intent: Dictionary containing user intent with required fields
        
    Raises:
        SummarizationError: If validation fails
    """
    if not isinstance(intent, dict):
        raise SummarizationError("Intent must be a dictionary")
    
    # Check if action is summarize
    action = intent.get("action")
    if action != "summarize":
        raise SummarizationError(f"Invalid action '{action}'. Expected 'summarize'")
    
    # Check if either file_id or query_text is present
    file_id = intent.get("file_id")
    query_text = intent.get("query_text")
    
    if not file_id and not query_text:
        raise SummarizationError("Either 'file_id' or 'query_text' must be provided")
    
    # Validate user_id is present
    user_id = intent.get("user_id")
    if not user_id:
        raise SummarizationError("'user_id' is required for summarization")
    
    logger.info(f"Intent validation passed for user {user_id}")


def determine_summarization_type(intent: Dict[str, Any]) -> str:
    """
    Determine the type of summarization based on the intent.
    
    Args:
        intent: Dictionary containing user intent
        
    Returns:
        String indicating summarization type: "file-based" or "query-based"
    """
    file_id = intent.get("file_id")
    query_text = intent.get("query_text")
    
    if file_id and not query_text:
        return "file-based"
    elif query_text and not file_id:
        return "query-based"
    elif file_id and query_text:
        # If both are present, prioritize file-based summarization with query context
        logger.info("Both file_id and query_text present, using file-based summarization with query context")
        return "file-based"
    else:
        # This should not happen due to validation, but handle it gracefully
        raise SummarizationError("Unable to determine summarization type")


def search_chunks_for_summarization(intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Search for document chunks using the search module functions.
    
    Args:
        intent: Dictionary containing user intent
        
    Returns:
        List of chunk dictionaries from search results
        
    Raises:
        SummarizationError: If search fails
    """
    user_id = intent["user_id"]
    summarization_type = determine_summarization_type(intent)
    
    # Set top_k based on summarization needs (more chunks for detailed summaries)
    summary_type = intent.get("summary_type", "short")
    top_k = 30 if summary_type == "detailed" else 15
    
    logger.info(f"Starting {summarization_type} search with top_k={top_k}")
    
    try:
        if summarization_type == "file-based":
            # Use search_by_file_id for file-based summarization
            file_id = intent["file_id"]
            query_text = intent.get("query_text", "")  # Optional query for context
            
            logger.info(f"Searching within file {file_id} for user {user_id}")
            chunks = search_by_file_id(
                file_id=file_id,
                user_id=user_id,
                query_text=query_text,
                top_k=top_k
            )
            
        else:
            # Use search_documents for query-based summarization
            query_text = intent["query_text"]
            
            # Create search intent using the helper function
            search_intent = create_search_intent(
                query_text=query_text,
                user_id=user_id,
                platform=intent.get("platform"),
                file_type=intent.get("file_type"),
                time_range=intent.get("time_range")
            )
            
            logger.info(f"Searching documents for query: '{query_text}'")
            chunks = search_documents(search_intent, top_k=top_k)
        
        logger.info(f"Found {len(chunks)} chunks from search")
        return chunks
        
    except SearchError as e:
        logger.error(f"Search error: {str(e)}")
        raise SummarizationError(f"Failed to search documents: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}")
        raise SummarizationError(f"Unexpected error during document search: {str(e)}")


def extract_file_name(chunks: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract file name from chunks metadata for file-based summarization.
    
    Args:
        chunks: List of chunk dictionaries containing metadata
        
    Returns:
        File name if found, None otherwise
    """
    if not chunks:
        return None
    
    # Try to extract file name from the first chunk's metadata
    first_chunk = chunks[0]
    
    # Check various possible metadata fields for file name
    metadata_fields = ["fileName", "file_name", "filename", "name", "title"]
    
    for field in metadata_fields:
        if field in first_chunk and first_chunk[field]:
            return first_chunk[field]
    
    # If no file name found in metadata, try to extract from file_id or other fields
    if "file_id" in first_chunk:
        return f"Document_{first_chunk['file_id']}"
    elif "fileId" in first_chunk:
        return f"Document_{first_chunk['fileId']}"
    
    return None


def prepare_chunks_for_summary(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Prepare chunks for summarization by extracting text content.
    
    Args:
        chunks: List of chunk dictionaries from search results
        
    Returns:
        List of text strings ready for summarization
    """
    text_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Extract text content from various possible fields
        text_content = None
        
        # Try different field names for chunk content (based on search.py structure)
        content_fields = ["text", "content", "chunk_text", "body", "data"]
        
        for field in content_fields:
            if field in chunk and chunk[field]:
                text_content = chunk[field]
                break
        
        if text_content:
            # Clean and prepare the text
            cleaned_text = str(text_content).strip()
            if cleaned_text:
                text_chunks.append(cleaned_text)
            else:
                logger.warning(f"Empty text content after cleaning in chunk {i}")
        else:
            logger.warning(f"No text content found in chunk {i}: {list(chunk.keys())}")
    
    logger.info(f"Prepared {len(text_chunks)} text chunks for summarization")
    return text_chunks


def validate_chunks_quality(chunks: List[Dict[str, Any]], intent: Dict[str, Any]) -> None:
    """
    Validate that the retrieved chunks are suitable for summarization.
    
    Args:
        chunks: List of chunk dictionaries
        intent: Original intent dictionary
        
    Raises:
        SummarizationError: If chunks are not suitable for summarization
    """
    if not chunks:
        summarization_type = determine_summarization_type(intent)
        if summarization_type == "file-based":
            error_msg = f"No content found for file_id: {intent.get('file_id')}"
        else:
            error_msg = f"No relevant content found for query: {intent.get('query_text')}"
        raise SummarizationError(error_msg)
    
    # Check if chunks have reasonable similarity scores (if available)
    scored_chunks = [c for c in chunks if '_similarity' in c and c['_similarity'] is not None]
    if scored_chunks:
        avg_similarity = sum(c['_similarity'] for c in scored_chunks) / len(scored_chunks)
        if avg_similarity < 0.3:  # Very low similarity threshold
            logger.warning(f"Low average similarity score: {avg_similarity:.3f}")
    
    # Check total content length
    total_content_length = 0
    for chunk in chunks:
        text_content = chunk.get('text') or chunk.get('content', '')
        total_content_length += len(str(text_content))
    
    if total_content_length < 100:  # Minimum content length
        raise SummarizationError("Retrieved content is too short for meaningful summarization")
    
    logger.info(f"Chunks validation passed: {len(chunks)} chunks, {total_content_length} total characters")


def create_summary_context(intent: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create context information for the summary.
    
    Args:
        intent: Original intent dictionary
        chunks: Retrieved chunks
        
    Returns:
        Dictionary with context information
    """
    context = {
        "summarization_type": determine_summarization_type(intent),
        "chunks_count": len(chunks),
        "user_query": intent.get("query_text"),
        "summary_type": intent.get("summary_type", "short")
    }
    
    # Add file information for file-based summarization
    if context["summarization_type"] == "file-based":
        context["file_id"] = intent.get("file_id")
        context["file_name"] = extract_file_name(chunks)
    
    # Add platform and time range if specified
    if intent.get("platform"):
        context["platform"] = intent["platform"]
    if intent.get("time_range"):
        context["time_range"] = intent["time_range"]
    
    # Calculate average similarity if available
    scored_chunks = [c for c in chunks if '_similarity' in c and c['_similarity'] is not None]
    if scored_chunks:
        context["average_similarity"] = sum(c['_similarity'] for c in scored_chunks) / len(scored_chunks)
    
    return context


def summarize_document(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to perform document summarization based on user intent.
    
    Args:
        intent: Dictionary containing structured user intent with fields:
            - action: Must be "summarize"
            - file_id: Optional file identifier for file-based summarization
            - query_text: Optional query text for query-based summarization
            - summary_type: Optional summary detail level ("short" or "detailed")
            - user_id: Required user identifier
            - platform: Optional platform identifier
            - file_type: Optional file type filter
            - time_range: Optional time range filter
            
    Returns:
        Dictionary containing:
            - summary: Generated summary text
            - summary_type: Type of summary generated
            - chunks_used: Number of chunks used for summarization
            - fileName: File name (only for file-based summarization)
            - context: Additional context information
            
    Raises:
        SummarizationError: If summarization fails
    """
    try:
        # Validate input intent
        validate_intent(intent)
        
        # Set default summary type if not specified
        summary_type = intent.get("summary_type", "short")
        if summary_type not in ["short", "detailed"]:
            logger.warning(f"Invalid summary_type '{summary_type}', defaulting to 'short'")
            summary_type = "short"
            intent["summary_type"] = summary_type
        
        # Determine summarization type
        summarization_type = determine_summarization_type(intent)
        
        logger.info(f"Starting {summarization_type} summarization for user {intent['user_id']}")
        
        # Search for relevant chunks using the search module
        try:
            chunks = search_chunks_for_summarization(intent)
        except Exception as e:
            logger.error(f"Error searching for chunks: {str(e)}")
            raise SummarizationError(f"Failed to retrieve document chunks: {str(e)}")
        
        # Validate chunks quality
        validate_chunks_quality(chunks, intent)
        
        # Prepare chunks for summarization
        text_chunks = prepare_chunks_for_summary(chunks)
        
        if not text_chunks:
            raise SummarizationError("No valid text content found in retrieved chunks")
        
        # Create context for summary
        context = create_summary_context(intent, chunks)
        
        # Generate summary using OpenAI
        try:
            query_for_summary = intent.get("query_text") if summarization_type == "query-based" else None
            summary = summarize_chunks(text_chunks, summary_type, query_for_summary)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise SummarizationError(f"Failed to generate summary: {str(e)}")
        
        # Prepare response
        response = {
            "summary": summary,
            "summary_type": summary_type,
            "chunks_used": len(text_chunks),
            "context": context
        }
        
        # Add file name for file-based summarization
        if summarization_type == "file-based":
            file_name = extract_file_name(chunks)
            if file_name:
                response["fileName"] = file_name
        
        # Add search metadata if available
        if chunks and '_search_metadata' in chunks[0]:
            response["search_metadata"] = chunks[0]['_search_metadata']
        
        logger.info(f"Successfully generated {summary_type} summary using {len(text_chunks)} chunks")
        
        return response
        
    except SummarizationError:
        # Re-raise custom errors
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in summarization: {str(e)}")
        raise SummarizationError(f"Unexpected error during summarization: {str(e)}")


def summarize_search_results(search_results: List[Dict[str, Any]], 
                           summary_type: str = "short",
                           query_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Summarize pre-existing search results without performing a new search.
    This function allows for a two-step process: search first, then summarize.
    
    Args:
        search_results: List of search result dictionaries from search_documents
        summary_type: Type of summary ("short" or "detailed")
        query_context: Optional query context for focused summarization
        
    Returns:
        Dictionary containing summarization results
        
    Raises:
        SummarizationError: If summarization fails
    """
    try:
        if not search_results:
            raise SummarizationError("No search results provided for summarization")
        
        if summary_type not in ["short", "detailed"]:
            logger.warning(f"Invalid summary_type '{summary_type}', defaulting to 'short'")
            summary_type = "short"
        
        logger.info(f"Summarizing {len(search_results)} pre-existing search results")
        
        # Prepare chunks for summarization
        text_chunks = prepare_chunks_for_summary(search_results)
        
        if not text_chunks:
            raise SummarizationError("No valid text content found in search results")
        
        # Generate summary using OpenAI
        try:
            summary = summarize_chunks(text_chunks, summary_type, query_context)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise SummarizationError(f"Failed to generate summary: {str(e)}")
        
        # Create context information
        context = {
            "summarization_type": "search-results-based",
            "chunks_count": len(search_results),
            "text_chunks_used": len(text_chunks),
            "summary_type": summary_type
        }
        
        # Add query context if provided
        if query_context:
            context["query_context"] = query_context
        
        # Calculate average similarity if available
        scored_results = [r for r in search_results if '_similarity' in r and r['_similarity'] is not None]
        if scored_results:
            context["average_similarity"] = sum(r['_similarity'] for r in scored_results) / len(scored_results)
        
        # Extract unique file names
        file_names = set()
        for result in search_results:
            file_name = result.get('fileName') or result.get('file_name')
            if file_name:
                file_names.add(file_name)
        
        if file_names:
            context["source_files"] = list(file_names)
        
        response = {
            "summary": summary,
            "summary_type": summary_type,
            "chunks_used": len(text_chunks),
            "context": context
        }
        
        logger.info(f"Successfully generated {summary_type} summary from search results")
        
        return response
        
    except SummarizationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error summarizing search results: {str(e)}")
        raise SummarizationError(f"Unexpected error during search results summarization: {str(e)}")


def summarize_file(file_id: str, user_id: str, summary_type: str = "short", 
                  platform: Optional[str] = None, query_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for file-based summarization.
    
    Args:
        file_id: Identifier of the file to summarize
        user_id: User identifier
        summary_type: Type of summary ("short" or "detailed")
        platform: Optional platform identifier
        query_context: Optional query for focused summarization within the file
        
    Returns:
        Dictionary containing summarization results
    """
    intent = {
        "action": "summarize",
        "file_id": file_id,
        "user_id": user_id,
        "summary_type": summary_type
    }
    
    if platform:
        intent["platform"] = platform
    
    if query_context:
        intent["query_text"] = query_context
    
    return summarize_document(intent)


def summarize_query(query_text: str, user_id: str, summary_type: str = "short",
                   platform: Optional[str] = None, file_type: Optional[str] = None,
                   time_range: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for query-based summarization.
    
    Args:
        query_text: Query text to search and summarize
        user_id: User identifier
        summary_type: Type of summary ("short" or "detailed")
        platform: Optional platform identifier
        file_type: Optional file type filter
        time_range: Optional time range filter
        
    Returns:
        Dictionary containing summarization results
    """
    intent = {
        "action": "summarize",
        "query_text": query_text,
        "user_id": user_id,
        "summary_type": summary_type
    }
    
    if platform:
        intent["platform"] = platform
    
    if file_type:
        intent["file_type"] = file_type
    
    if time_range:
        intent["time_range"] = time_range
    
    return summarize_document(intent)


# Workflow functions for search-then-summarize pattern
def search_and_summarize(search_intent: Dict[str, Any], summary_type: str = "short") -> Dict[str, Any]:
    """
    Combined workflow: search documents then summarize the results.
    This enables a two-step process where search results can be reviewed before summarization.
    
    Args:
        search_intent: Dictionary for search_documents function
        summary_type: Type of summary to generate
        
    Returns:
        Dictionary containing both search results and summary
    """
    try:
        # Validate search intent
        from .search import validate_search_intent
        is_valid, error_msg = validate_search_intent(search_intent)
        if not is_valid:
            raise SummarizationError(f"Invalid search intent: {error_msg}")
        
        logger.info(f"Starting search-and-summarize workflow for user {search_intent.get('user_id')}")
        
        # Perform search
        try:
            search_results = search_documents(search_intent, top_k=20)
        except SearchError as e:
            raise SummarizationError(f"Search failed: {str(e)}")
        
        if not search_results:
            raise SummarizationError("No search results found for summarization")
        
        # Summarize the search results
        query_context = search_intent.get('query_text')
        summary_result = summarize_search_results(search_results, summary_type, query_context)
        
        # Combine results
        combined_result = {
            "search_results": search_results,
            "summary": summary_result["summary"],
            "summary_type": summary_result["summary_type"],
            "chunks_used": summary_result["chunks_used"],
            "context": summary_result["context"],
            "workflow_type": "search_and_summarize"
        }
        
        logger.info(f"Successfully completed search-and-summarize workflow")
        
        return combined_result
        
    except SummarizationError:
        raise
    except Exception as e:
        logger.error(f"Error in search-and-summarize workflow: {str(e)}")
        raise SummarizationError(f"Search-and-summarize workflow failed: {str(e)}")


if __name__ == "__main__":
    # Sample usage and testing
    
    # Example 1: File-based summarization
    print("=== File-based Summarization Test ===")
    try:
        file_intent = {
            "action": "summarize",
            "file_id": "doc_12345",
            "user_id": "user_67890",
            "summary_type": "short",
            "platform": "google_drive"
        }
        
        result = summarize_document(file_intent)
        print(f"File Summary Result: {result}")
        
    except SummarizationError as e:
        print(f"File summarization error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Example 2: Query-based summarization
    print("\n=== Query-based Summarization Test ===")
    try:
        query_intent = {
            "action": "summarize",
            "query_text": "What are the main findings about AI in healthcare?",
            "user_id": "user_67890",
            "summary_type": "detailed",
            "platform": "google_drive"
        }
        
        result = summarize_document(query_intent)
        print(f"Query Summary Result: {result}")
        
    except SummarizationError as e:
        print(f"Query summarization error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Example 3: Search-and-summarize workflow
    print("\n=== Search-and-Summarize Workflow Test ===")
    try:
        search_intent = {
            "query_text": "machine learning algorithms",
            "user_id": "user_12345",
            "platform": "google_drive",
            "time_range": "last_30_days"
        }
        
        result = search_and_summarize(search_intent, "detailed")
        print(f"Search-and-summarize result: {len(result['search_results'])} results, summary length: {len(result['summary'])}")
        
    except SummarizationError as e:
        print(f"Search-and-summarize error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Example 4: Using convenience functions
    print("\n=== Convenience Functions Test ===")
    try:
        # Test file summarization convenience function
        file_result = summarize_file("doc_98765", "user_12345", "short", "onedrive")
        print(f"Convenience file summary: {file_result}")
        
        # Test query summarization convenience function
        query_result = summarize_query("machine learning trends", "user_12345", "detailed")
        print(f"Convenience query summary: {query_result}")
        
    except Exception as e:
        print(f"Convenience function error: {e}")
    
    print("\nAll tests completed!")