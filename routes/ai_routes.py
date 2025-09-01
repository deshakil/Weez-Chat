# ai_layer/tools.py

from typing import List, Dict, Any
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
    FileCategory
)
from ai_layer.summarizer import summarize_document
from ai_layer.rag import answer_query_with_rag

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
    """Enhanced summarize tool wrapper with comprehensive debugging."""
    try:
        # DEBUGGING: Log the complete structure of received arguments
        logger.info("=" * 60)
        logger.info("DEBUGGING: tool_summarize received arguments")
        logger.info("=" * 60)
        logger.info(f"Raw args type: {type(args)}")
        logger.info(f"Raw args keys: {list(args.keys()) if isinstance(args, dict) else 'Not a dict'}")

        # Pretty print the entire args structure
        logger.info("Complete args structure:")
        try:
            logger.info(json.dumps(args, indent=2, default=str))
        except Exception as json_error:
            logger.warning(f"JSON serialization failed: {json_error}")
            logger.info(f"Args via pprint: {pprint.pformat(args)}")
            logger.info(f"Args via str: {str(args)}")

        # Check each key individually
        for key, value in args.items():
            logger.info(f"Key '{key}': {type(value)} = {repr(value)}")

        # Specifically examine file_id
        file_id = args.get("file_id")
        logger.info(f"file_id extracted: type={type(file_id)}, value={repr(file_id)}")

        # Check if file_id might be nested in another structure
        if not file_id:
            logger.info("file_id not found at top level, checking for nested structures...")
            for key, value in args.items():
                if isinstance(value, dict):
                    logger.info(f"Checking nested dict in key '{key}': {value}")
                    if "file_id" in value:
                        logger.info(f"Found file_id in nested key '{key}': {value['file_id']}")
                elif isinstance(value, list):
                    logger.info(f"Checking list in key '{key}': {value}")
                    for i, item in enumerate(value):
                        if isinstance(item, dict) and "file_id" in item:
                            logger.info(f"Found file_id in list item {i}: {item['file_id']}")

        logger.info("=" * 60)

        # Ensure required 'action' field is present
        if "action" not in args:
            args["action"] = "summarize"

        # Validate required parameters
        user_id = args.get("user_id")
        if not user_id:
            return {
                "error": "Missing required parameter: user_id",
                "tool_used": "summarize"
            }

        # Enhanced file_id extraction with multiple fallback strategies
        file_id = None

        # Strategy 1: Direct access
        file_id = args.get("file_id")
        if file_id:
            logger.info(f"Strategy 1 SUCCESS: file_id = {file_id}")

        # Strategy 2: Check for common nested patterns
        if not file_id:
            # Check if it's nested in a 'file' or 'document' object
            for nested_key in ["file", "document", "doc", "item"]:
                if nested_key in args and isinstance(args[nested_key], dict):
                    nested_file_id = args[nested_key].get("id") or args[nested_key].get("file_id")
                    if nested_file_id:
                        file_id = nested_file_id
                        logger.info(f"Strategy 2 SUCCESS: file_id found in {nested_key}: {file_id}")
                        break

        # Strategy 3: Check for ID patterns in any string values
        if not file_id:
            for key, value in args.items():
                if isinstance(value, str) and key.lower() in ["id", "document_id", "doc_id"]:
                    file_id = value
                    logger.info(f"Strategy 3 SUCCESS: file_id found in {key}: {file_id}")
                    break

        # Strategy 4: Look for numeric or UUID-like patterns
        if not file_id:
            for key, value in args.items():
                if isinstance(value, (str, int)):
                    # Check if it looks like an ID (numeric or UUID pattern)
                    str_value = str(value)
                    if (str_value.isdigit() and len(str_value) <= 10) or \
                            (len(str_value) == 36 and str_value.count('-') == 4):
                        logger.info(f"Strategy 4 POTENTIAL: Found ID-like value in {key}: {value}")
                        # Don't auto-assign, but log for manual inspection

        query_text = args.get("query_text", "")

        # More detailed logging about what we found
        logger.info(f"Final extracted values:")
        logger.info(f"  - file_id: {repr(file_id)} (type: {type(file_id)})")
        logger.info(f"  - query_text: {repr(query_text)} (type: {type(query_text)})")
        logger.info(f"  - user_id: {repr(user_id)} (type: {type(user_id)})")

        if not file_id and not query_text:
            return {
                "error": "Either file_id or query_text must be provided",
                "tool_used": "summarize",
                "debug_info": {
                    "received_args": args,
                    "extraction_attempts": "All strategies failed to find file_id"
                }
            }

        # Set default summary type if not provided
        if "summary_type" not in args:
            args["summary_type"] = "short"

        # Validate summary_type
        valid_summary_types = ["short", "detailed", "bullet", "executive", "technical"]
        if args["summary_type"] not in valid_summary_types:
            args["summary_type"] = "short"

        # Log final args being passed to summarize_document
        final_args = dict(args)  # Create a copy
        if file_id:
            final_args["file_id"] = file_id

        logger.info(f"Final args being passed to summarize_document: {json.dumps(final_args, indent=2, default=str)}")

        result = summarize_document(final_args)

        # Ensure result has proper structure
        if not isinstance(result, dict):
            result = {"summary": str(result)}

        # Add tool metadata
        result["tool_used"] = "summarize"

        return result

    except Exception as e:
        logger.info(f"Raw args type: {type(args)}")
        logger.info(f"Raw args keys: {list(args.keys()) if isinstance(args, dict) else 'Not a dict'}")
        logger.error(f"Summarization error: {str(e)}", exc_info=True)
        return {
            "error": f"Summarization failed: {str(e)}",
            "tool_used": "summarize",
            "debug_info": {
                "received_args": args if 'args' in locals() else "args not available",
                "extracted_file_id": file_id if 'file_id' in locals() else "file_id not extracted"
            }
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
            "description": "Summarize a document by file ID or based on a query with various summary types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string", 
                        "description": "The ID of the file to summarize"
                    },
                    "query_text": {
                        "type": "string", 
                        "description": "Optional query for query-based summarization"
                    },
                    "summary_type": {
                        "type": "string", 
                        "description": "Type of summary to generate",
                        "enum": ["short", "detailed", "bullet", "executive", "technical"]
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "The user ID"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
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
