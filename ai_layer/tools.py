# ai_layer/tools.py

from typing import List, Dict, Any, Optional
import logging
import json
import pprint

# Import from the new search.py module
from ai_layer.search import (
    search_documents,
    search_by_file_id,
    get_similar_documents,
    get_search_suggestions,
    get_search_stats,
    create_search_intent,
    validate_search_intent,
    SearchError,
    Platform,
    FileCategory,
    search_files_by_name
)
from ai_layer.summarizer import summarize_document
from ai_layer.rag import answer_query_with_rag
from utils.cosmos_client import create_cosmos_client

# Configure logger
logger = logging.getLogger(__name__)

# ===========================
# 🔍 Search Tool Wrapper
# ===========================
def tool_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced search tool wrapper using the new search.py module.
    Supports all advanced features including validation, filtering, and result processing.
    """
    try:
        # Extract required parameters
        logger.info(f"[DEBUG] tool_search received args: {args}")
        query_text = args.get("query_text")
        user_id = args.get("user_id")

        if not query_text or not user_id:
            return {
                "error": "Missing required parameters: query_text and user_id",
                "tool_used": "search"
            }

        print(args)

        # Extract optional parameters with proper validation
        top_k = args.get("top_k", 10)

        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            top_k = 10  # Default fallback

        # Build search intent using the create_search_intent helper
        intent_kwargs = {}

        # Map platform parameter (convert from old format if needed)
        if args.get("platform"):
            platform = args["platform"].lower().strip()
            # Map common platform names to search.py format
            platform_mapping = {
                "drive": "google_drive",
                "google_drive": "google_drive",
                "onedrive": "onedrive",
                "dropbox": "dropbox",
                "sharepoint": "sharepoint",
                "local": "local",
                "slack": "slack",
                "teams": "teams"
            }
            intent_kwargs["platform"] = platform_mapping.get(platform, platform)

        # Map mime_type/file_type parameter
        if args.get("mime_type"):
            # Convert MIME type to search.py file_type format
            mime_type = args["mime_type"]
            mime_to_file_type = {
                'application/pdf': 'PDF',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
                'application/msword': 'DOC',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLSX',
                'application/vnd.ms-excel': 'XLS',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PPTX',
                'application/vnd.ms-powerpoint': 'PPT',
                'text/plain': 'TXT'
            }
            intent_kwargs["file_type"] = mime_to_file_type.get(mime_type, mime_type)
        elif args.get("file_type"):
            intent_kwargs["file_type"] = args["file_type"]

        # Map time_range parameter
        if args.get("time_range"):
            time_range = args["time_range"]
            if isinstance(time_range, dict):
                # Convert from/to format to start_date/end_date format
                converted_time_range = {}
                if time_range.get("from"):
                    converted_time_range["start_date"] = time_range["from"]
                if time_range.get("to"):
                    converted_time_range["end_date"] = time_range["to"]
                intent_kwargs["time_range"] = converted_time_range
            elif isinstance(time_range, str):
                # Handle relative time ranges
                intent_kwargs["time_range"] = time_range

        # Add pagination parameters if provided
        if args.get("offset"):
            intent_kwargs["offset"] = args["offset"]
        if args.get("limit"):
            intent_kwargs["limit"] = min(args["limit"], 100)  # Cap at 100

        # Create search intent
        intent = create_search_intent(query_text, user_id, **intent_kwargs)

        # Validate the intent
        is_valid, error_message = validate_search_intent(intent)
        if not is_valid:
            return {
                "error": f"Invalid search intent: {error_message}",
                "tool_used": "search"
            }

        # Perform the search
        logger.info(f"Search intent created: {intent}")
        logger.info(f"Executing search for user {user_id} with query: '{query_text}'")
        results = search_documents(intent, top_k=top_k)

        # Extract applied filters for response
        applied_filters = {}
        if intent.get("platform"):
            applied_filters["platform"] = intent["platform"]
        if intent.get("file_type"):
            applied_filters["file_type"] = intent["file_type"]
        if intent.get("time_range"):
            applied_filters["time_range"] = intent["time_range"]

        # Format response
        response = {
            "tool_used": "search",
            "query_text": query_text,
            "user_id": user_id,
            "filters_applied": applied_filters,
            "total_results": len(results),
            "results": results
        }

        # Add search metrics if available (from first result)
        if results and "_search_metrics" in results[0]:
            response["search_metrics"] = results[0]["_search_metrics"]

        logger.info(f"Search completed: {len(results)} results returned")
        return response

    except SearchError as e:
        logger.error(f"Search error: {str(e)}")
        return {
            "error": f"Search failed: {str(e)}",
            "tool_used": "search"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_search: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected search error: {str(e)}",
            "tool_used": "search"
        }


# ===========================
# 🔍 File-Specific Search Tool
# ===========================
def tool_search_file(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search within a specific file using the search_by_file_id function.
    """
    try:
        file_id = args.get("file_id")
        user_id = args.get("user_id")
        query_text = args.get("query_text", "")
        top_k = args.get("top_k", 10)

        if not file_id or not user_id:
            return {
                "error": "Missing required parameters: file_id and user_id",
                "tool_used": "search_file"
            }

        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            top_k = 10

        logger.info(f"Searching within file {file_id} for user {user_id}")
        results = search_by_file_id(file_id, user_id, query_text, top_k)

        return {
            "tool_used": "search_file",
            "file_id": file_id,
            "user_id": user_id,
            "query_text": query_text,
            "total_results": len(results),
            "results": results
        }

    except SearchError as e:
        logger.error(f"File search error: {str(e)}")
        return {
            "error": f"File search failed: {str(e)}",
            "tool_used": "search_file"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_search_file: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected file search error: {str(e)}",
            "tool_used": "search_file"
        }


# ===========================
# 🔍 Similar Documents Tool
# ===========================
def tool_similar_documents(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find documents similar to a given file.
    """
    try:
        file_id = args.get("file_id")
        user_id = args.get("user_id")
        top_k = args.get("top_k", 5)

        if not file_id or not user_id:
            return {
                "error": "Missing required parameters: file_id and user_id",
                "tool_used": "similar_documents"
            }

        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            top_k = 5

        logger.info(f"Finding documents similar to {file_id} for user {user_id}")
        results = get_similar_documents(file_id, user_id, top_k)

        return {
            "tool_used": "similar_documents",
            "source_file_id": file_id,
            "user_id": user_id,
            "total_results": len(results),
            "similar_documents": results
        }

    except SearchError as e:
        logger.error(f"Similar documents error: {str(e)}")
        return {
            "error": f"Similar documents search failed: {str(e)}",
            "tool_used": "similar_documents"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_similar_documents: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected similar documents error: {str(e)}",
            "tool_used": "similar_documents"
        }


# ===========================
# 🔍 Search Suggestions Tool
# ===========================
def tool_search_suggestions(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate search suggestions based on partial query.
    """
    try:
        partial_query = args.get("partial_query", "")
        user_id = args.get("user_id")
        limit = args.get("limit", 5)

        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "search_suggestions"
            }

        if len(partial_query.strip()) < 2:
            return {
                "tool_used": "search_suggestions",
                "partial_query": partial_query,
                "suggestions": []
            }

        # Validate limit
        if not isinstance(limit, int) or limit < 1 or limit > 20:
            limit = 5

        logger.debug(f"Generating search suggestions for '{partial_query}' for user {user_id}")
        suggestions = get_search_suggestions(partial_query, user_id, limit)

        return {
            "tool_used": "search_suggestions",
            "partial_query": partial_query,
            "user_id": user_id,
            "total_suggestions": len(suggestions),
            "suggestions": suggestions
        }

    except Exception as e:
        logger.error(f"Error generating search suggestions: {str(e)}")
        return {
            "tool_used": "search_suggestions",
            "partial_query": args.get("partial_query", ""),
            "suggestions": []
        }


# ===========================
# 📊 Search Stats Tool
# ===========================
def tool_search_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get search and document statistics for a user.
    """
    try:
        user_id = args.get("user_id")

        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "search_stats"
            }

        logger.info(f"Retrieving search statistics for user {user_id}")
        stats = get_search_stats(user_id)

        return {
            "tool_used": "search_stats",
            "user_id": user_id,
            "statistics": stats
        }

    except SearchError as e:
        logger.error(f"Search stats error: {str(e)}")
        return {
            "error": f"Failed to retrieve search statistics: {str(e)}",
            "tool_used": "search_stats"
        }
    except Exception as e:
        logger.error(f"Unexpected error in tool_search_stats: {str(e)}", exc_info=True)
        return {
            "error": f"Unexpected error retrieving statistics: {str(e)}",
            "tool_used": "search_stats"
        }


# ===========================
# 🧠 Summarize Tool Wrapper
# ===========================
def tool_summarize(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced summarize tool wrapper. Brain.py now handles most file resolution logic.
    """
    try:
        user_id = args.get("user_id")
        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "summarize"
            }

        # Check if we have file_id (preferred) or fall back to other parameters
        file_id = args.get("file_id")
        query_text = args.get("query_text", "")
        if file_id:
            # Check if it looks like a fileName (has extension, spaces, etc.)
            import re
            if (re.search(r'\.(pdf|docx?|xlsx?|pptx?|txt|csv)$', file_id.lower()) or
                    ' ' in file_id or '(' in file_id or ')' in file_id):
                logger.error(f"❌ CRITICAL: Received fileName as file_id: {file_id}")
                return {
                    "error": f"Invalid file_id format. Received what appears to be a fileName: '{file_id}'. This should have been resolved to a proper file_id.",
                    "tool_used": "summarize",
                    "debug_info": {
                        "received_file_id": file_id,
                        "issue": "fileName passed as file_id"
                    }
                }

        # Try filename parameters if no file_id
        if not file_id:
            filename_candidates = [
                args.get("filename"),
                args.get("file_name"),
                args.get("fileName"),
                args.get("name")
            ]

            found_filename = None
            for candidate in filename_candidates:
                if candidate and isinstance(candidate, str) and candidate.strip():
                    found_filename = candidate.strip()
                    # Try to resolve filename to file_id
                    try:
                        from ai_layer.search import search_files_by_name
                        results = search_files_by_name(
                            file_name=found_filename,
                            user_id=user_id,
                            exact_match=False,
                            limit=1
                        )
                        if results and len(results) > 0:
                            file_id = results[0].get('file_id')
                            if file_id:
                                args["file_id"] = file_id
                                logger.info(f"Resolved filename '{found_filename}' to file_id: {file_id}")
                                break
                    except Exception as resolve_error:
                        logger.error(f"Failed to resolve filename: {resolve_error}")
                    break

        if not file_id and not query_text:
            return {
                "error": "Could not resolve file reference. Need either file_id, filename, or query_text.",
                "tool_used": "summarize",
                "suggestions": [
                    "Use the search tool first to find the file you want to summarize",
                    "Provide a more specific filename",
                    "Use query-based summarization with descriptive query_text"
                ]
            }

        # Set defaults
        if "summary_type" not in args:
            args["summary_type"] = "short"
        if "action" not in args:
            args["action"] = "summarize"

        logger.info(f"Calling summarize_document with file_id: {file_id}")
        result = summarize_document(args)

        if not isinstance(result, dict):
            result = {"summary": str(result)}

        result["tool_used"] = "summarize"
        result["file_id_used"] = file_id

        return result

    except Exception as e:
        logger.error(f"Summarization error: {str(e)}", exc_info=True)
        return {
            "error": f"Summarization failed: {str(e)}",
            "tool_used": "summarize"
        }
# ===========================
# 📖 RAG Tool Wrapper
# ===========================
def tool_rag(args: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced RAG tool wrapper with proper argument formatting."""
    try:
        # Ensure required 'action' field is present if needed
        if "action" not in args:
            args["action"] = "rag"

        # Validate required parameters
        user_id = args.get("user_id")
        query_text = args.get("query_text")

        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "rag"
            }

        if not query_text:
            return {
                "error": "Missing required parameter: query_text",
                "tool_used": "rag"
            }

        # Set default max_context_chunks if not provided
        if "max_context_chunks" not in args:
            args["max_context_chunks"] = 5

        # Validate max_context_chunks
        max_chunks = args["max_context_chunks"]
        if not isinstance(max_chunks, int) or max_chunks < 1 or max_chunks > 20:
            args["max_context_chunks"] = 5

        logger.info(f"Calling answer_query_with_rag with args: {args}")
        result = answer_query_with_rag(args)

        # Ensure result has proper structure
        if not isinstance(result, dict):
            result = {"answer": str(result)}

        # Add tool metadata
        result["tool_used"] = "rag"

        return result

    except Exception as e:
        logger.error(f"RAG error: {str(e)}", exc_info=True)
        return {
            "error": f"RAG query failed: {str(e)}",
            "tool_used": "rag"
        }


# ===========================
# 🛠 Enhanced Tool Function Registry
# ===========================
TOOL_FUNCTIONS = {
    "search": {
        "function": tool_search,
        "spec": {
            "name": "search",
            "description": "Search relevant document chunks using hybrid vector + metadata search with advanced filtering capabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The user's search query (2-1000 characters)",
                        "minLength": 2,
                        "maxLength": 1000
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID for data isolation"
                    },
                    "platform": {
                        "type": "string",
                        "description": "Platform filter: google_drive, onedrive, dropbox, sharepoint, local, slack, teams",
                        "enum": ["google_drive", "onedrive", "dropbox", "sharepoint", "local", "slack", "teams"]
                    },
                    "file_type": {
                        "type": "string",
                        "description": "File type filter: PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, TXT",
                        "enum": ["PDF", "DOCX", "DOC", "XLSX", "XLS", "PPTX", "PPT", "TXT"]
                    },
                    "mime_type": {
                        "type": "string",
                        "description": "MIME type filter (alternative to file_type)"
                    },
                    "time_range": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "Relative time range",
                                "enum": ["last_hour", "last_24_hours", "last_7_days", "last_30_days", "last_month", "last_3_months", "last_6_months", "last_year"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string", "format": "date", "description": "Start date (ISO format)"},
                                    "to": {"type": "string", "format": "date", "description": "End date (ISO format)"}
                                },
                                "description": "Absolute time range filter"
                            }
                        ]
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return (1-100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset",
                        "minimum": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Pagination limit (1-100)",
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query_text", "user_id"]
            }
        }
    },
    "search_file": {
        "function": tool_search_file,
        "spec": {
            "name": "search_file",
            "description": "Search within a specific document by file ID using semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Unique identifier for the file to search within"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID for data isolation"
                    },
                    "query_text": {
                        "type": "string",
                        "description": "Optional search query within the file",
                        "default": ""
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (1-100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    }
                },
                "required": ["file_id", "user_id"]
            }
        }
    },
    "similar_documents": {
        "function": tool_similar_documents,
        "spec": {
            "name": "similar_documents",
            "description": "Find documents similar to a given file using semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Source file ID to find similar documents for"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID for data isolation"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar documents to return (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["file_id", "user_id"]
            }
        }
    },
    "search_suggestions": {
        "function": tool_search_suggestions,
        "spec": {
            "name": "search_suggestions",
            "description": "Generate search suggestions based on partial query input and user's document collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "partial_query": {
                        "type": "string",
                        "description": "Partial search query to generate suggestions for"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID for personalized suggestions"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of suggestions to return (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    "search_stats": {
        "function": tool_search_stats,
        "spec": {
            "name": "search_stats",
            "description": "Get comprehensive search and document statistics for a user's collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to get statistics for"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    "summarize": {
        "function": tool_summarize,
        "spec": {
            "name": "summarize",
            "description": "Summarize a document by file ID, filename, or based on a query with various summary types. Supports advanced filename resolution with fuzzy matching and confidence scoring. At least one of: file_id, filename, file_name, fileName, name, or query_text must be provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID for document access and filename resolution"
                    },
                    "file_id": {
                        "type": "string",
                        "description": "The ID of the file to summarize (takes precedence over filename)"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to summarize (will be resolved to file_id using advanced search)"
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Alternative parameter name for filename"
                    },
                    "fileName": {
                        "type": "string",
                        "description": "Alternative parameter name for filename"
                    },
                    "name": {
                        "type": "string",
                        "description": "Alternative parameter name for filename"
                    },
                    "query_text": {
                        "type": "string",
                        "description": "Query for query-based summarization (used when file_id/filename not provided)"
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "Type of summary to generate",
                        "enum": ["short", "detailed", "bullet", "executive", "technical"],
                        "default": "short"
                    },
                    "platform": {
                        "type": "string",
                        "description": "Optional platform filter for filename resolution",
                        "enum": ["google_drive", "onedrive", "dropbox", "sharepoint", "local", "slack", "teams",
                                 "notion"]
                    }
                },
                "required": ["user_id"]
            }
        }
    }
    ,
    "rag": {
        "function": tool_rag,
        "spec": {
            "name": "rag",
            "description": "Answer a question using RAG (retrieval-augmented generation) over user documents with context retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The user's question to answer using document context"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user ID for document access"
                    },
                    "max_context_chunks": {
                        "type": "integer",
                        "description": "Maximum number of document chunks to use as context (1-20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["query_text", "user_id"]
            }
        }
    }
}


# ===========================
# 🔧 Utility Functions
# ===========================
def get_file_id_by_name_enhanced(file_name: str, user_id: str, platform: Optional[str] = None,
                                 confidence_threshold: float = 0.5) -> Optional[Dict[str, Any]]:
    """
    Enhanced file ID resolution with confidence scoring.
    This is a placeholder implementation - you may need to implement the actual logic.
    """
    try:
        # Use the existing search_files_by_name function as a fallback
        results = search_files_by_name(
            file_name=file_name,
            user_id=user_id,
            platform=platform,
            exact_match=False,
            limit=1
        )

        if results and len(results) > 0:
            result = results[0]
            return {
                'file_id': result.get('file_id'),
                'confidence': result.get('search_relevance', 0.7),  # Assume decent confidence
                'strategy': 'basic_search_enhanced',
                'fileName': result.get('fileName'),
                'platform': result.get('platform')
            }

        return None
    except Exception as e:
        logger.error(f"Enhanced file ID resolution failed: {e}")
        return None


def search_files_by_name_advanced(file_name_query: str, user_id: str, exact_match: bool = False,
                                  include_fuzzy: bool = True, filters: Optional[Dict[str, Any]] = None,
                                  limit: int = 5) -> List[Dict[str, Any]]:
    """
    Advanced file search with multiple strategies.
    This is a placeholder implementation - you may need to implement the actual logic.
    """
    try:
        # Use the existing search_files_by_name function as a fallback
        platform = None
        if filters and 'platforms' in filters and filters['platforms']:
            platform = filters['platforms'][0]  # Use first platform

        results = search_files_by_name(
            file_name=file_name_query,
            user_id=user_id,
            platform=platform,
            exact_match=exact_match,
            limit=limit
        )

        # Add metadata for advanced search
        for result in results:
            result['_final_relevance'] = result.get('search_relevance', 0.5)
            result['_search_strategy'] = 'advanced_fallback'

        return results
    except Exception as e:
        logger.error(f"Advanced file search failed: {e}")
        return []



def diagnose_file_search_issue(user_id: str, file_name: str) -> Dict[str, Any]:
    """
    Diagnostic function to help identify why file search/summarization is failing.
    """
    try:
        cosmos_client = create_cosmos_client()

        diagnostic_info = {
            "user_id": user_id,
            "file_name": file_name,
            "searches_performed": [],
            "found_files": [],
            "chunk_analysis": {}
        }

        # 1. Search for exact filename matches
        query1 = """
            SELECT DISTINCT c.file_id, c.fileId, c.fileName, COUNT(1) as chunk_count
            FROM c 
            WHERE c.user_id = @user_id 
            AND LOWER(c.fileName) = LOWER(@file_name)
            GROUP BY c.file_id, c.fileId, c.fileName
        """

        results1 = list(cosmos_client.container.query_items(
            query=query1,
            parameters=[
                {"name": "@user_id", "value": user_id},
                {"name": "@file_name", "value": file_name}
            ],
            enable_cross_partition_query=True
        ))

        diagnostic_info["searches_performed"].append({
            "type": "exact_filename_match",
            "query": query1,
            "results_count": len(results1),
            "results": results1
        })

        # 2. Search for partial filename matches
        query2 = """
            SELECT DISTINCT c.file_id, c.fileId, c.fileName, COUNT(1) as chunk_count
            FROM c 
            WHERE c.user_id = @user_id 
            AND CONTAINS(LOWER(c.fileName), LOWER(@partial_name))
            GROUP BY c.file_id, c.fileId, c.fileName
        """

        # Extract core filename without extension
        partial_name = file_name.replace('.pdf', '').replace('_removed', '').replace(' ', '').replace('(', '').replace(
            ')', '')

        results2 = list(cosmos_client.container.query_items(
            query=query2,
            parameters=[
                {"name": "@user_id", "value": user_id},
                {"name": "@partial_name", "value": partial_name}
            ],
            enable_cross_partition_query=True
        ))

        diagnostic_info["searches_performed"].append({
            "type": "partial_filename_match",
            "partial_name_used": partial_name,
            "query": query2,
            "results_count": len(results2),
            "results": results2
        })

        # 3. Get all files for this user (limited sample)
        query3 = """
            SELECT DISTINCT TOP 20 c.file_id, c.fileId, c.fileName, c.platform, c.mime_type
            FROM c 
            WHERE c.user_id = @user_id 
            AND IS_DEFINED(c.fileName)
        """

        results3 = list(cosmos_client.container.query_items(
            query=query3,
            parameters=[{"name": "@user_id", "value": user_id}],
            enable_cross_partition_query=True
        ))

        diagnostic_info["searches_performed"].append({
            "type": "all_user_files_sample",
            "query": query3,
            "results_count": len(results3),
            "results": results3
        })

        # 4. Analyze chunk structure for found files
        all_found_files = results1 + results2
        for file_info in all_found_files:
            file_id = file_info.get('file_id') or file_info.get('fileId')
            if file_id:
                # Get sample chunks for this file
                chunk_query = """
                    SELECT TOP 5 c.id, c.chunk_index, c.text, c.content, c.embedding
                    FROM c 
                    WHERE c.user_id = @user_id 
                    AND (c.file_id = @file_id OR c.fileId = @file_id)
                """

                chunk_results = list(cosmos_client.container.query_items(
                    query=chunk_query,
                    parameters=[
                        {"name": "@user_id", "value": user_id},
                        {"name": "@file_id", "value": file_id}
                    ],
                    enable_cross_partition_query=True
                ))

                diagnostic_info["chunk_analysis"][file_id] = {
                    "chunk_count": len(chunk_results),
                    "chunks": [
                        {
                            "id": chunk.get("id"),
                            "chunk_index": chunk.get("chunk_index"),
                            "has_text": bool(chunk.get("text")),
                            "has_content": bool(chunk.get("content")),
                            "has_embedding": bool(chunk.get("embedding")),
                            "text_length": len(chunk.get("text", "")),
                            "content_length": len(chunk.get("content", ""))
                        }
                        for chunk in chunk_results
                    ]
                }

        return diagnostic_info

    except Exception as e:
        return {
            "error": f"Diagnostic failed: {str(e)}",
            "user_id": user_id,
            "file_name": file_name
        }

def get_available_platforms() -> List[str]:
    """Get list of supported platforms."""
    return [platform.value for platform in Platform]


def get_available_file_categories() -> List[str]:
    """Get list of supported file categories."""
    return [category.value for category in FileCategory]


def get_tool_names() -> List[str]:
    """Get list of available tool names."""
    return list(TOOL_FUNCTIONS.keys())


def get_tool_spec(tool_name: str) -> Dict[str, Any]:
    """Get the specification for a specific tool."""
    if tool_name in TOOL_FUNCTIONS:
        return TOOL_FUNCTIONS[tool_name]["spec"]
    return {}


# ===========================
# 📝 Export Information
# ===========================
__all__ = [
    'TOOL_FUNCTIONS',
    'tool_search',
    'tool_search_file',
    'tool_similar_documents',
    'tool_search_suggestions',
    'tool_search_stats',
    'tool_summarize',
    'tool_rag',
    'get_available_platforms',
    'get_available_file_categories',
    'get_tool_names',
    'get_tool_spec'
]