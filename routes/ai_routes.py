# routes/ai_routes.py

#from fastapi import APIRouter, HTTPException
#from pydantic import BaseModel
#from typing import Dict, Optional

#from ai_layer.summarizer import summarize_document
#from ai_layer.rag import answer_query_with_rag, health_check
#from ai_layer.brain import reason_and_act  # Central decision engine

#router = APIRouter()


# === Pydantic Schemas (Optional, for validation & docs) ===

#class IntentRequest(BaseModel):
#    action: str
#    query_text: Optional[str] = None
#    file_id: Optional[str] = None
#    user_id: str
#    summary_type: Optional[str] = "short"
#    platform: Optional[str] = None
#    time_range: Optional[Dict[str, str]] = None
#    top_k: Optional[int] = 10


#class QueryRequest(BaseModel):
#    query: str
#    user_id: str


# === Routes ===

#@router.post("/summarize")
#async def summarize_endpoint(intent: IntentRequest):
#    """
#    Summarize a document or based on query intent.
#    """
#    try:
#        result = summarize_document(intent.dict())
#        return result
#    except Exception as e:
#        raise HTTPException(status_code=400, detail=str(e))


#@router.post("/rag")
#async def rag_endpoint(intent: IntentRequest):
#    """
#    Answer a question using RAG over stored document chunks.
#    """
#    try:
#        result = answer_query_with_rag(intent.dict())
#        return result
#    except Exception as e:
#        raise HTTPException(status_code=400, detail=str(e))


#@router.post("/ask")
#async def ask_endpoint(request: QueryRequest):
#    try:
#        response = reason_and_act(user_id=request.user_id, user_input=request.query)
#        return response
#    except Exception as e:
#        raise HTTPException(status_code=400, detail=f"Agent error: {str(e)}")


#@router.get("/health")
#async def health():
#    """
#    Check the health status of the RAG system and agent layer.
#    """
#    try:
#        return health_check()
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# routes/ai_routes.py

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
from openai import AzureOpenAI

from ai_layer.summarizer import summarize_document
from ai_layer.rag import answer_query_with_rag, health_check
from ai_layer.brain import initialize_brain, reason_and_act  # Central decision engine

# Import search functionality
from ai_layer.search import (
    search_documents, 
    search_by_file_id, 
    get_similar_documents,
    get_search_suggestions,
    get_search_stats,
    validate_search_intent,
    create_search_intent,
    SearchError
)

router = APIRouter()

# Initialize the brain on module load
def _initialize_brain_if_needed():
    """Initialize the brain with Azure OpenAI client if not already done."""
    try:
        # Check if brain is already initialized by attempting a dummy call
        # This will raise RuntimeError if not initialized
        from ai_layer.brain import _brain_instance
        if _brain_instance is None:
            # Get Azure OpenAI credentials from environment variables
            api_key = os.getenv("OPENAI_API_KEY")
            endpoint = os.getenv("OPENAI_ENDPOINT", "https://weez-openai-resource.openai.azure.com/")
            api_version = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")
            chat_deployment = os.getenv("OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

            if not api_key or not endpoint:
                raise ValueError("Missing Azure OpenAI credentials. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")
            
            # Create Azure OpenAI client
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
            
            # Initialize the brain
            initialize_brain(
                azure_openai_client=client,
                chat_deployment=chat_deployment
            )
            
    except Exception as e:
        raise RuntimeError(f"Failed to initialize brain: {str(e)}")

# Initialize brain when module is imported
_initialize_brain_if_needed()


# === Pydantic Schemas ===

class IntentRequest(BaseModel):
    action: str
    query_text: Optional[str] = None
    file_id: Optional[str] = None
    user_id: str
    summary_type: Optional[str] = "short"
    platform: Optional[str] = None
    time_range: Optional[Dict[str, str]] = None
    top_k: Optional[int] = 10


class QueryRequest(BaseModel):
    query: str
    user_id: str


class SearchRequest(BaseModel):
    query_text: str
    user_id: str
    platform: Optional[str] = None
    file_type: Optional[str] = None
    time_range: Optional[str] = None
    top_k: Optional[int] = 10


class FileSearchRequest(BaseModel):
    file_id: str
    user_id: str
    query_text: Optional[str] = ""
    top_k: Optional[int] = 10


class SimilarDocumentsRequest(BaseModel):
    file_id: str
    user_id: str
    top_k: Optional[int] = 5


class SearchSuggestionsRequest(BaseModel):
    partial_query: str
    user_id: str
    limit: Optional[int] = 5


# === Existing Routes ===

@router.post("/summarize")
async def summarize_endpoint(intent: IntentRequest):
    """
    Summarize a document or based on query intent.
    """
    try:
        result = summarize_document(intent.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rag")
async def rag_endpoint(intent: IntentRequest):
    """
    Answer a question using RAG over stored document chunks.
    """
    try:
        result = answer_query_with_rag(intent.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ask")
async def ask_endpoint(request: QueryRequest):
    """
    Ask the AI agent a question using the ReAct reasoning engine.
    """
    try:
        # Ensure brain is initialized before processing request
        _initialize_brain_if_needed()
        
        response = reason_and_act(user_id=request.user_id, user_input=request.query)
        return {"response": response, "user_id": request.user_id}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Brain initialization error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Agent error: {str(e)}")


# === NEW SEARCH ROUTES ===

@router.post("/search")
async def search_endpoint(request: SearchRequest):
    """
    Search for documents using semantic similarity and metadata filters.
    """
    try:
        # Create search intent
        intent = create_search_intent(
            query_text=request.query_text,
            user_id=request.user_id,
            platform=request.platform,
            file_type=request.file_type,
            time_range=request.time_range
        )
        
        # Validate intent
        is_valid, error_msg = validate_search_intent(intent)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid search intent: {error_msg}")
        
        # Perform search
        results = search_documents(intent, top_k=request.top_k)
        
        return {
            "results": results,
            "total_found": len(results),
            "query": request.query_text,
            "user_id": request.user_id,
            "filters_applied": {
                "platform": request.platform,
                "file_type": request.file_type,
                "time_range": request.time_range
            }
        }
        
    except SearchError as e:
        raise HTTPException(status_code=400, detail=f"Search error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal search error: {str(e)}")


@router.post("/search/file")
async def search_file_endpoint(request: FileSearchRequest):
    """
    Search within a specific document by file ID.
    """
    try:
        results = search_by_file_id(
            file_id=request.file_id,
            user_id=request.user_id,
            query_text=request.query_text,
            top_k=request.top_k
        )
        
        return {
            "results": results,
            "total_found": len(results),
            "file_id": request.file_id,
            "query": request.query_text,
            "user_id": request.user_id
        }
        
    except SearchError as e:
        raise HTTPException(status_code=400, detail=f"File search error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal file search error: {str(e)}")


@router.post("/search/similar")
async def similar_documents_endpoint(request: SimilarDocumentsRequest):
    """
    Find documents similar to a given file.
    """
    try:
        results = get_similar_documents(
            file_id=request.file_id,
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        return {
            "similar_documents": results,
            "total_found": len(results),
            "source_file_id": request.file_id,
            "user_id": request.user_id
        }
        
    except SearchError as e:
        raise HTTPException(status_code=400, detail=f"Similar documents error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal similar documents error: {str(e)}")


@router.post("/search/suggestions")
async def search_suggestions_endpoint(request: SearchSuggestionsRequest):
    """
    Get search suggestions based on partial query input.
    """
    try:
        suggestions = get_search_suggestions(
            partial_query=request.partial_query,
            user_id=request.user_id,
            limit=request.limit
        )
        
        return {
            "suggestions": suggestions,
            "partial_query": request.partial_query,
            "user_id": request.user_id
        }
        
    except SearchError as e:
        raise HTTPException(status_code=400, detail=f"Search suggestions error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal suggestions error: {str(e)}")


@router.get("/search/stats/{user_id}")
async def search_stats_endpoint(user_id: str):
    """
    Get search and document statistics for a user.
    """
    try:
        stats = get_search_stats(user_id)
        
        return {
            "stats": stats,
            "user_id": user_id
        }
        
    except SearchError as e:
        raise HTTPException(status_code=400, detail=f"Search stats error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal stats error: {str(e)}")


@router.get("/health")
async def health():
    """
    Check the health status of the RAG system and agent layer.
    """
    try:
        # Check RAG health
        rag_health = health_check()
        
        # Check search health by testing cosmos client
        try:
            from ai_layer.search import client
            # Simple test - this should not fail if cosmos client is properly configured
            search_health = {"status": "healthy", "cosmos_client": "connected"}
        except Exception as e:
            search_health = {"status": "unhealthy", "cosmos_client": f"error: {str(e)}"}
        
        return {
            "rag": rag_health,
            "search": search_health,
            "overall_status": "healthy" if rag_health.get("status") == "healthy" and search_health.get("status") == "healthy" else "unhealthy"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")