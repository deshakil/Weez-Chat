"""
FastAPI routes for conversation management using CosmosMemoryManager
These routes connect your React frontend to your Cosmos DB backend
Fixed version addressing ConnectionResetError issues and brain initialization
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime
from pydantic import BaseModel
from urllib.parse import unquote
import asyncio

# Add missing imports for brain initialization
from openai import AzureOpenAI
from ai_layer.brain import reason_and_act, initialize_brain

# Import your memory manager
from ai_layer.memory import CosmosMemoryManager, CosmosMemoryError

# Configure logging with more specific formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress asyncio error logging for connection resets (Windows specific)
asyncio_logger = logging.getLogger('asyncio')
asyncio_logger.setLevel(logging.WARNING)

app = FastAPI(
    title="Weez AI Conversation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add trusted host middleware first
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production, specify exact hosts
)

# Enhanced CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=False,  # Set to False to avoid complex CORS
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)

# Initialize memory manager (you might want to use dependency injection in production)
memory_manager = CosmosMemoryManager()

# Global flag to track brain initialization
_brain_initialized = False


# Pydantic models for request/response
class StoreConversationRequest(BaseModel):
    user_query: str
    agent_response: str
    timestamp: Optional[str] = None


class ConversationMessage(BaseModel):
    id: str
    user_id: str
    conversation_id: str
    user_query: str
    agent_response: str
    timestamp: str


class ConversationSummary(BaseModel):
    conversation_id: str
    first_message_time: str
    last_message_time: str
    message_count: int
    latest_user_query: Optional[str] = None
    latest_agent_response: Optional[str] = None


class SearchRequest(BaseModel):
    search_term: str
    conversation_id: Optional[str] = None
    limit: Optional[int] = 10


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    user_id: str


# Custom exception handler for connection errors
@app.exception_handler(ConnectionError)
async def connection_error_handler(request: Request, exc: ConnectionError):
    logger.warning(f"Connection error occurred: {exc}")
    return {"error": "Connection issue", "detail": "Please try again"}


# Dependency to get memory manager
def get_memory_manager() -> CosmosMemoryManager:
    return memory_manager


def _initialize_brain_if_needed():
    """Initialize the brain with Azure OpenAI client if not already done."""
    global _brain_initialized

    if _brain_initialized:
        return

    try:
        # Get Azure OpenAI credentials from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("OPENAI_ENDPOINT", "https://weez-openai-resource.openai.azure.com/")
        api_version = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")
        chat_deployment = os.getenv("OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

        if not api_key:
            raise ValueError("Missing Azure OpenAI API key. Please set OPENAI_API_KEY environment variable.")

        if not endpoint:
            raise ValueError("Missing Azure OpenAI endpoint. Please set OPENAI_ENDPOINT environment variable.")

        logger.info(f"Initializing brain with endpoint: {endpoint}")
        logger.info(f"Using chat deployment: {chat_deployment}")

        # Create Azure OpenAI client
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

        # Initialize the brain
        initialize_brain(
            azure_openai_client=client,
            chat_deployment=chat_deployment,
            memory_manager=memory_manager
        )

        _brain_initialized = True
        logger.info("Brain initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize brain: {str(e)}")
        raise RuntimeError(f"Failed to initialize brain: {str(e)}")


# Initialize brain at startup
@app.on_event("startup")
async def startup_event():
    """Initialize the brain when the application starts."""
    try:
        _initialize_brain_if_needed()
    except Exception as e:
        logger.error(f"Failed to initialize brain at startup: {e}")
        # Don't fail startup, but log the error


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        health_status = memory_manager.health_check()
        brain_status = "initialized" if _brain_initialized else "not_initialized"
        return {
            "status": "healthy",
            "cosmos_db": health_status,
            "brain": brain_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.options("/api/conversations/{user_id}")
async def options_user_conversations(user_id: str):
    """Handle CORS preflight for user conversations endpoint."""
    return {"status": "ok"}


@app.get("/api/conversations/{user_id}", response_model=List[ConversationSummary])
async def get_user_conversations(
        user_id: str,
        request: Request,
        limit: Optional[int] = None,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Get all conversation summaries for a user.
    This corresponds to memory.get_user_conversations()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        logger.info(f"Getting conversations for user: {user_id}")

        # Check if client is still connected before processing
        if await request.is_disconnected():
            logger.warning(f"Client disconnected before processing conversations for {user_id}")
            raise HTTPException(status_code=499, detail="Client disconnected")

        conversations = memory.get_user_conversations(user_id, limit)

        # Convert to response format
        result = []
        for conv in conversations:
            # Check connection periodically during processing
            if len(result) % 50 == 0 and await request.is_disconnected():
                logger.warning(f"Client disconnected during processing for {user_id}")
                raise HTTPException(status_code=499, detail="Client disconnected")

            result.append(ConversationSummary(
                conversation_id=conv["conversation_id"],
                first_message_time=conv["first_message_time"],
                last_message_time=conv["last_message_time"],
                message_count=conv["message_count"],
                latest_user_query=conv.get("latest_user_query"),
                latest_agent_response=conv.get("latest_agent_response")
            ))

        logger.info(f"Successfully retrieved {len(result)} conversations for user {user_id}")
        return result

    except HTTPException:
        raise
    except CosmosMemoryError as e:
        logger.error(f"CosmosMemoryError for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except ValueError as e:
        logger.error(f"ValueError for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error for user {user_id}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.options("/api/conversations/{user_id}/{conversation_id}")
async def options_conversation_history(user_id: str, conversation_id: str):
    """Handle CORS preflight for conversation history endpoint."""
    return {"status": "ok"}


@app.get("/api/conversations/{user_id}/{conversation_id}", response_model=List[ConversationMessage])
async def get_conversation_history(
        user_id: str,
        conversation_id: str,
        request: Request,
        limit: Optional[int] = None,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Get full conversation history for a specific conversation.
    This corresponds to memory.get_conversation_history()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversation_id = unquote(conversation_id)  # Decode URL-encoded conversation_id

        # Check if client is still connected
        if await request.is_disconnected():
            logger.warning(f"Client disconnected before processing conversation {conversation_id}")
            raise HTTPException(status_code=499, detail="Client disconnected")

        messages = memory.get_conversation_history(user_id, conversation_id, limit)

        # Convert to response format
        result = []
        for msg in messages:
            result.append(ConversationMessage(
                id=msg["id"],
                user_id=msg["user_id"],
                conversation_id=msg["conversation_id"],
                user_query=msg["user_query"],
                agent_response=msg["agent_response"],
                timestamp=msg["timestamp"]
            ))

        logger.info(f"Retrieved {len(result)} messages for conversation {conversation_id}")
        return result

    except HTTPException:
        raise
    except CosmosMemoryError as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/conversations/{user_id}/{conversation_id}")
async def store_conversation(
        user_id: str,
        conversation_id: str,
        request: StoreConversationRequest,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Store a new conversation message.
    This corresponds to memory.store_conversation()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversation_id = unquote(conversation_id)  # Decode URL-encoded conversation_id

        memory.store_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            user_query=request.user_query,
            agent_response=request.agent_response,
            timestamp=request.timestamp
        )

        logger.info(f"Stored conversation for user {user_id} in conversation {conversation_id}")
        return {"status": "success", "message": "Conversation stored successfully"}

    except CosmosMemoryError as e:
        logger.error(f"Failed to store conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/conversations/{user_id}/{conversation_id}")
async def delete_conversation(
        user_id: str,
        conversation_id: str,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Delete an entire conversation thread.
    This corresponds to memory.delete_conversation()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversation_id = unquote(conversation_id)  # Decode URL-encoded conversation_id

        deleted_count = memory.delete_conversation(user_id, conversation_id)

        logger.info(f"Deleted {deleted_count} messages from conversation {conversation_id}")
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} messages",
            "deleted_count": deleted_count
        }

    except CosmosMemoryError as e:
        logger.error(f"Failed to delete conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/conversations/{user_id}/search")
async def search_conversations(
        user_id: str,
        request: SearchRequest,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Search through conversation history.
    This corresponds to memory.search_conversations()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id

        results = memory.search_conversations(
            user_id=user_id,
            search_term=request.search_term,
            conversation_id=request.conversation_id,
            limit=request.limit
        )

        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(ConversationMessage(
                id=result["id"],
                user_id=result["user_id"],
                conversation_id=result["conversation_id"],
                user_query=result["user_query"],
                agent_response=result["agent_response"],
                timestamp=result["timestamp"]
            ))

        logger.info(f"Found {len(search_results)} search results for '{request.search_term}'")
        return {
            "results": search_results,
            "total_count": len(search_results),
            "search_term": request.search_term
        }

    except CosmosMemoryError as e:
        logger.error(f"Failed to search conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/conversations/{user_id}/{conversation_id}/context")
async def get_conversation_context(
        user_id: str,
        conversation_id: str,
        context_limit: int = 5,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Get recent conversation context for AI prompts.
    This corresponds to memory.get_recent_context()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversation_id = unquote(conversation_id)  # Decode URL-encoded conversation_id

        context = memory.get_recent_context(user_id, conversation_id, context_limit)

        return {
            "conversation_id": conversation_id,
            "context": context,
            "context_limit": context_limit
        }

    except CosmosMemoryError as e:
        logger.error(f"Failed to get conversation context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/conversations/{user_id}/cleanup")
async def cleanup_old_conversations(
        user_id: str,
        days_old: int = 90,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Clean up old conversations.
    This corresponds to memory.cleanup_old_conversations()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id

        deleted_count = memory.cleanup_old_conversations(user_id, days_old)

        logger.info(f"Cleaned up {deleted_count} old messages for user {user_id}")
        return {
            "status": "success",
            "message": f"Cleaned up {deleted_count} old messages",
            "deleted_count": deleted_count,
            "days_old": days_old
        }

    except CosmosMemoryError as e:
        logger.error(f"Failed to cleanup conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# AI Agent Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(
        request: ChatRequest,
        user_id: str,  # You might get this from authentication/headers
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Main chat endpoint that integrates with the AI brain.
    This is the missing piece that connects your frontend to the AI agent.
    """
    try:
        # Ensure brain is initialized before processing
        _initialize_brain_if_needed()

        # Input validation
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message is required")

        logger.info(f"Chat request from user {user_id}: {request.message[:100]}...")
        logger.info(f"Conversation ID: {request.conversation_id}")

        # Call the AI brain with the conversation_id (can be None for new conversations)
        ai_response = reason_and_act(
            user_id=user_id,
            user_input=request.message,
            conversation_id=request.conversation_id
        )

        # If this was a new conversation, the brain would have generated an ID
        # We need to get it from the stored conversation
        final_conversation_id = request.conversation_id
        if not final_conversation_id:
            # Get the latest conversation for this user to find the new conversation_id
            try:
                recent_conversations = memory.get_user_conversations(user_id, limit=1)
                if recent_conversations:
                    final_conversation_id = recent_conversations[0]['conversation_id']
            except Exception as e:
                logger.error(f"Failed to get conversation ID: {e}")
                final_conversation_id = "unknown"

        logger.info(f"AI response generated for conversation {final_conversation_id}")

        return ChatResponse(
            response=ai_response,
            conversation_id=final_conversation_id,
            user_id=user_id
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/api/chat/{user_id}", response_model=ChatResponse)
async def chat_with_ai_user_param(
        user_id: str,
        request: ChatRequest,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """Chat endpoint with user_id as path parameter and proper conversation handling."""
    try:
        # Ensure brain is initialized before processing
        _initialize_brain_if_needed()

        user_id = unquote(user_id)  # Decode URL-encoded user_id

        # Input validation
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message is required")

        logger.info(f"Chat request from user {user_id}: {request.message[:100]}...")
        logger.info(f"Conversation ID: {request.conversation_id}")

        # Validate conversation_id exists if provided
        if request.conversation_id:
            try:
                # Check if conversation exists
                existing_history = memory.get_conversation_history(
                    user_id=user_id,
                    conversation_id=request.conversation_id,
                    limit=1
                )
                if existing_history:
                    logger.info(
                        f"Continuing existing conversation {request.conversation_id} with {len(existing_history)} messages")
                else:
                    logger.info(f"Conversation {request.conversation_id} not found, treating as new")
            except Exception as e:
                logger.warning(f"Could not validate conversation {request.conversation_id}: {e}")

        # Call the AI brain
        ai_response = reason_and_act(
            user_id=user_id,
            user_input=request.message,
            conversation_id=request.conversation_id
        )

        # Use the conversation_id that was passed or generated
        final_conversation_id = request.conversation_id
        if not final_conversation_id:
            # If it was a new conversation, get the latest one
            try:
                recent_conversations = memory.get_user_conversations(user_id, limit=1)
                if recent_conversations:
                    final_conversation_id = recent_conversations[0]['conversation_id']
            except Exception as e:
                logger.error(f"Failed to get conversation ID: {e}")
                final_conversation_id = "unknown"

        logger.info(f"AI response generated for conversation {final_conversation_id}")

        return ChatResponse(
            response=ai_response,
            conversation_id=final_conversation_id,
            user_id=user_id
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


# Debug endpoint
@app.get("/api/conversations/{user_id}/{conversation_id}/debug")
async def debug_conversation_context(
        user_id: str,
        conversation_id: str,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """Debug endpoint to check conversation history and context."""
    try:
        user_id = unquote(user_id)
        conversation_id = unquote(conversation_id)

        # Get conversation history
        history = memory.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=10
        )

        # Get recent context
        context = memory.get_recent_context(
            user_id=user_id,
            conversation_id=conversation_id,
            context_limit=5
        )

        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "history_count": len(history),
            "history": history,
            "formatted_context": context,
            "debug_info": {
                "conversation_exists": len(history) > 0,
                "last_message_time": history[-1]["timestamp"] if history else None,
                "brain_initialized": _brain_initialized
            }
        }

    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(CosmosMemoryError)
async def cosmos_memory_error_handler(request, exc: CosmosMemoryError):
    """Handle Cosmos DB memory errors."""
    logger.error(f"Cosmos Memory Error: {str(exc)}")
    return HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    # Enhanced uvicorn configuration for Windows
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info",
        # Windows-specific configurations
        loop="asyncio",  # Use asyncio loop explicitly
        http="httptools",  # Use httptools for better performance
        ws_ping_interval=20,  # WebSocket ping interval
        ws_ping_timeout=20,  # WebSocket ping timeout
        timeout_keep_alive=30,  # Keep alive timeout
        limit_concurrency=1000,  # Limit concurrent connections
        limit_max_requests=10000,  # Max requests before restarting
        backlog=2048,  # Connection backlog
    )