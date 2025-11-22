"""
brain.py — Enterprise-grade ReAct-style reasoning agent for Weezy MCP AI Agent (Azure OpenAI).

This module coordinates:
  • Intent parsing (`intent_parser.parse_intent`)
  • Embedding generation (`embedder.get_query_embedding`)
  • Tool execution: search, summarize, rag
  • Per-user conversational + semantic memory (`CosmosMemoryManager`)
  • Azure OpenAI Chat Completions with *function calling* (aka tools) in a ReAct loop

Design goals
============
• **Deterministic orchestration**: Brain decides when to delegate vs directly tool-call.
• **Model-led tool routing**: Uses ChatCompletions w/ tools schemas so the model can choose.
• **Multi-step ReAct**: Model may call multiple tools sequentially; loop limited by `max_reasoning_steps`.
• **Graceful clarification**: If `parse_intent` returns `needs_clarification`, we immediately ask user.
• **Memory augmentation**: Recent queries + tool results are injected as conversation context to improve grounding.
• **Robust error handling**: Captures tool errors, surfaces helpful fallback messages to the model and user.

Azure OpenAI Notes
==================
You must configure the `AzureOpenAI` client outside this module and pass it to `initialize_brain(...)`.
For Azure, the *model* argument to `chat.completions.create` is the **deployment name** you configured in Azure (e.g., "gpt-4o").

Message Protocol
================
We use the newer *tools* API structure:
  tools=[{"type":"function","function":{...}}]
Model responses may include `message.tool_calls` (a list). For each tool call we execute the mapped python function,
append a `role="tool"` message with the JSON result, then re-call the model.
We stop when model returns a message **without** tool calls or when `max_reasoning_steps` reached.

Usage
=====
>>> from openai import AzureOpenAI
>>> client = AzureOpenAI(api_key=..., api_version=..., azure_endpoint=...)
>>> initialize_brain(client, chat_deployment="gpt-4o")
>>> reply = reason_and_act(user_id="user123", user_input="Summarize yesterday's design meeting notes from Google Drive", conversation_id="conv-123")
print(reply)

"""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

from openai import AzureOpenAI

# --- Module Imports ---------------------------------------------------------
from .intent_parser import parse_user_intent
from .embedder import get_query_embedding

# tools.py is expected to expose TOOL_FUNCTIONS, a dict keyed by function name:
# {
#   "search": {"function": <callable>, "spec": {"name": "search", "description": "...", "parameters": {...}}},
#   "summarize": {"function": <callable>, "spec": {...}},
#   "rag": {"function": <callable>, "spec": {...}},
# }
from .tools import TOOL_FUNCTIONS

# Import the Cosmos DB memory manager
from .memory import CosmosMemoryManager

# ---------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: extract tool schemas + callables from TOOL_FUNCTIONS

def _extract_tool_schemas() -> List[Dict[str, Any]]:
    """Return a list of tool schema dicts in the format expected by Chat Completions."""
    schemas: List[Dict[str, Any]] = []
    for name, meta in TOOL_FUNCTIONS.items():
        # tools.py uses 'spec' key, not 'schema'
        spec = meta.get("spec")
        if not spec:
            logger.warning("Tool %s missing spec; skipping.", name)
            continue

        # Convert the spec to the format expected by Azure OpenAI
        schema = {
            "type": "function",
            "function": spec
        }
        schemas.append(schema)
    return schemas


def _extract_tool_callables() -> Dict[str, Callable[..., Any]]:
    mapping: Dict[str, Callable[..., Any]] = {}
    for name, meta in TOOL_FUNCTIONS.items():
        fn = meta.get("function")
        if callable(fn):
            mapping[name] = fn
        else:
            logger.warning("Tool %s has non-callable function entry.", name)
    return mapping


# ---------------------------------------------------------------------------
# ReAct Brain

class ReActBrain:
    """Reason + Act orchestration layer using Azure OpenAI function calling."""

    def __init__(
            self,
            azure_openai_client: AzureOpenAI,
            chat_deployment: str,
            memory_manager: Optional[CosmosMemoryManager] = None,
            max_reasoning_steps: int = 5,
            temperature: float = 0.1,
            conversation_history_limit: int = 10,
    ) -> None:
        self.client = azure_openai_client
        self.model = chat_deployment  # Azure deployment name
        self.memory_manager = memory_manager or CosmosMemoryManager()
        self.max_reasoning_steps = max_reasoning_steps
        self.temperature = temperature
        self.conversation_history_limit = conversation_history_limit

        self.tool_mapping = _extract_tool_callables()
        self.tool_schemas = _extract_tool_schemas()

    # ------------------------------------------------------------------
    # Public API
    def reason_and_act(self, user_id: str, user_input: str, conversation_id: Optional[str] = None) -> str:
        """Main entry point. Returns final user-facing response string."""
        logger.info("ReActBrain.start user=%s input=%s conversation_id=%s", user_id, user_input, conversation_id)

        # CRITICAL CHANGE: Only generate new conversation_id if none provided
        # This was causing context loss by creating new conversations
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info("Generated new conversation_id: %s", conversation_id)
            is_new_conversation = True
        else:
            logger.info("Using existing conversation_id: %s", conversation_id)
            is_new_conversation = False

        # Parse intent
        intent = self._safe_parse_intent(user_input)

        # Clarification gate
        if intent.get("needs_clarification"):
            clarification_msg = self._clarification_message(intent)
            # Store clarification request
            self._store_conversation(user_id, conversation_id, user_input, clarification_msg)
            return clarification_msg

        # Generate embedding
        query_text = intent.get("query_text") or user_input
        embedding = self._safe_get_embedding(query_text)

        # Build context - this will now properly include history for existing conversations
        messages = self._build_conversation_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            user_input=user_input,
            intent=intent,
            embedding=embedding
        )

        # ReAct loop
        reply = self._react_loop(user_id=user_id, conversation_id=conversation_id, messages=messages, intent=intent)

        # Store the complete conversation in memory
        self._store_conversation(user_id, conversation_id, user_input, reply)

        return reply

    # ------------------------------------------------------------------
    # Internal helpers
    def _safe_parse_intent(self, user_input: str) -> Dict[str, Any]:
        try:
            return parse_user_intent(user_input) or {}
        except Exception as e:  # fallback default minimal intent
            logger.error("Intent parsing failed: %s", e)
            logger.debug(traceback.format_exc())
            return {
                "action": "search",
                "query_text": user_input,
                "needs_clarification": False,
            }

    def _clarification_message(self, intent: Dict[str, Any]) -> str:
        reason = intent.get("clarification_reason")
        base = "I need a bit more information to help you effectively."
        if reason:
            base += f" {reason}"
        base += " Could you please clarify what you need (topic, file, platform, or format)?"
        return base

    def _safe_get_embedding(self, text: str) -> List[float]:
        try:
            return get_query_embedding(text)
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            logger.debug(traceback.format_exc())
            return []

    def _build_conversation_messages(
            self,
            user_id: str,
            conversation_id: str,
            user_input: str,
            intent: Dict[str, Any],
            embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Build conversation messages including system prompt and conversation history."""
        messages: List[Dict[str, Any]] = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": self._system_prompt(intent=intent, embedding=embedding)
        })

        # Get conversation history from Cosmos DB for this specific conversation
        try:
            # CRITICAL FIX: Always attempt to get history for existing conversations
            # The previous check was preventing history retrieval
            history = self.memory_manager.get_conversation_history(
                user_id=user_id,
                conversation_id=conversation_id,
                limit=self.conversation_history_limit
            )

            logger.info(f"Retrieved {len(history)} messages from conversation {conversation_id}")

            # Convert Cosmos DB history to chat format
            # History comes back in chronological order (ascending)
            for conversation in history:
                # Add user message
                user_msg = conversation.get("user_query", "").strip()
                if user_msg:
                    messages.append({
                        "role": "user",
                        "content": user_msg
                    })

                # Add assistant response
                assistant_msg = conversation.get("agent_response", "").strip()
                if assistant_msg:
                    messages.append({
                        "role": "assistant",
                        "content": assistant_msg
                    })

        except Exception as e:
            logger.error("Failed to retrieve conversation history: %s", e)
            logger.debug(traceback.format_exc())
            # Continue without history if retrieval fails

        # Add current user message
        messages.append({"role": "user", "content": user_input})

        logger.info(f"Built conversation with {len(messages)} total messages")
        return messages

    def _system_prompt(self, intent: Dict[str, Any], embedding: List[float]) -> str:
        """Generate system prompt with context."""
        prompt = (
            "You are Weez.AI MCP's enterprise AI reasoning agent. "
            "You have access to the full conversation history above. "
            "Use the available function tools to gather info (search, summarize, rag) when needed. "
            "Always consider the conversation context and refer back to previous messages when relevant. "
            "Think step-by-step: decide if you need to call a tool; if so, return a tool call. "
            "After tools return, synthesize a clear answer citing the tool results (do not hallucinate). "
            "Be helpful, accurate, and maintain conversational continuity.\n\n"

            # CRITICAL FIX: Add explicit instructions for file reference handling
            "IMPORTANT FILE HANDLING INSTRUCTIONS:\n"
            "- When users refer to files with phrases like 'this file', 'the above file', 'that document', etc., "
            "look for file_id values in recent search results from the conversation history.\n"
            "- For summarization tasks, ALWAYS use the file_id parameter, never use filename or file_name.\n"
            "- Search results contain file_id fields - extract these IDs when users want to act on specific files.\n"
            "- If a user asks to summarize 'this file' or 'the above file', find the file_id from the most recent search results.\n"
            "- Example: If search returned a file with 'file_id': 'abc123', use {'file_id': 'abc123'} for summarization.\n\n"
        )

        # Add lightweight context injection
        if intent:
            prompt += f"Intent context: action={intent.get('action')}, query={intent.get('query_text')}"
            if intent.get('platform'):
                prompt += f", platform={intent.get('platform')}"
            if intent.get('mime_type'):
                prompt += f", mime_type={intent.get('mime_type')}"
            prompt += "\n\n"

        if embedding:
            prompt += f"Embedding available (length: {len(embedding)}) for semantic search.\n\n"

        prompt += "Return responses in markdown format when appropriate."
        return prompt

    def _store_conversation(self, user_id: str, conversation_id: str, user_query: str, agent_response: str) -> None:
        """Store conversation in Cosmos DB with proper conversation threading."""
        try:
            self.memory_manager.store_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                user_query=user_query,
                agent_response=agent_response,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            logger.info("Stored conversation for user: %s, conversation: %s", user_id, conversation_id)
        except Exception as e:
            logger.error("Failed to store conversation: %s", e)
            logger.debug(traceback.format_exc())
            # Don't fail the entire response if storage fails

    # ------------------------------------------------------------------
    # Core ReAct loop w/ tool calling
    def _react_loop(self, user_id: str, conversation_id: str, messages: List[Dict[str, Any]],
                    intent: Dict[str, Any]) -> str:
        """Iteratively call the model; execute tool calls until done or step limit reached."""
        steps = 0
        while steps < self.max_reasoning_steps:
            steps += 1
            logger.debug("ReAct step %s messages_len=%s", steps, len(messages))

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error("Azure OpenAI chat call failed: %s", e)
                logger.debug(traceback.format_exc())
                return "I ran into an error contacting the language model. Please try again."

            msg = resp.choices[0].message

            # If the model returned tool calls, execute them
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                # Add the assistant message that triggered the tool calls
                messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        tc.model_dump() if hasattr(tc, "model_dump") else _tool_call_to_dict(tc)
                        for tc in tool_calls
                    ],
                })

                # Execute each tool call sequentially
                for tc in tool_calls:
                    name = (getattr(tc.function, "name", None) if hasattr(tc, "function")
                            else tc.get("function", {}).get("name"))
                    arg_str = (getattr(tc.function, "arguments", "{}") if hasattr(tc, "function")
                               else tc.get("function", {}).get("arguments", "{}"))
                    args = self._safe_json_loads(arg_str)

                    logger.info("Executing tool %s with args: %s", name, args)
                    result = self._dispatch_tool(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        tool_name=name,
                        args=args,
                        intent=intent,
                        messages=messages  # CRITICAL: Pass messages for context extraction
                    )

                    # Append tool result message
                    tool_call_id = (getattr(tc, "id", None) if hasattr(tc, "id")
                                    else tc.get("id"))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                # Continue loop to let model observe tool outputs
                continue

            # No tool calls => final answer
            content = msg.content or "I don't have further information."
            logger.info("ReActBrain.final steps=%s", steps)
            return content.strip()

        # Step limit hit; ask model to summarize
        logger.warning("ReActBrain reached max_reasoning_steps=%s; forcing finalization.", self.max_reasoning_steps)
        try:
            messages.append({
                "role": "user",
                "content": "Please provide your best final answer based on all tool results so far."
            })
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return (resp.choices[0].message.content or "(No response)").strip()
        except Exception as e:
            logger.error("Finalization call failed: %s", e)
            logger.debug(traceback.format_exc())
            return "I've gathered information but couldn't generate a final response. Please retry."

    # ------------------------------------------------------------------
    # Tool dispatch + error safety
    def _dispatch_tool(
            self,
            user_id: str,
            conversation_id: str,
            tool_name: Optional[str],
            args: Dict[str, Any],
            intent: Dict[str, Any],
            messages: List[Dict[str, Any]]  # Messages for context extraction
    ) -> Dict[str, Any]:
        """Execute a tool function with error handling and context-aware processing."""
        if not tool_name:
            return {"success": False, "error": "Missing tool name."}

        fn = self.tool_mapping.get(tool_name)
        if not fn:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        # CRITICAL FIX: Always override user_id from the authenticated user
        # The model may hallucinate or provide incorrect user_id values
        original_user_id = args.get("user_id")
        args["user_id"] = user_id  # Force override with authenticated user

        # Log the override for debugging if there was a mismatch
        if original_user_id and original_user_id != user_id:
            logger.warning("Model provided incorrect user_id '%s', overriding with authenticated user: %s",
                           original_user_id, user_id)

        # Augment args with intent defaults if missing
        if "platform" not in args and intent.get("platform") is not None:
            args["platform"] = intent["platform"]
        if "mime_type" not in args and intent.get("mime_type") is not None:
            args["mime_type"] = intent["mime_type"]

        # Tool-specific argument handling
        if tool_name == "summarize":
            # CRITICAL FIX: Always resolve fileName to file_id before passing to tools
            if not args.get("file_id"):
                # First try to extract from recent search results in conversation
                context_result = self._extract_file_id_from_context(messages)

                if context_result:
                    # Always check if it's a fileName and resolve it to file_id
                    if self._looks_like_file_id(context_result):
                        args["file_id"] = context_result
                        logger.info(f"Using file_id directly from context: {context_result}")
                    else:
                        # It's a fileName - MUST resolve to file_id before proceeding
                        logger.info(f"Context returned fileName: {context_result}, resolving to file_id...")
                        resolved_file_id = self._resolve_filename_to_file_id(user_id, context_result)
                        if resolved_file_id:
                            args["file_id"] = resolved_file_id
                            logger.info(
                                f"✅ Successfully resolved fileName '{context_result}' to file_id: {resolved_file_id}")
                        else:
                            logger.error(f"❌ FAILED to resolve fileName '{context_result}' to file_id")
                            return {
                                "success": False,
                                "function": tool_name,
                                "error": f"Could not resolve fileName '{context_result}' to file_id. File may not exist or be accessible."
                            }
                else:
                    # Fallback: Try filename-based resolution from direct args
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
                            break

                    if found_filename:
                        # MUST resolve filename to file_id
                        logger.info(f"Attempting to resolve filename '{found_filename}' to file_id...")
                        resolved_file_id = self._resolve_filename_to_file_id(user_id, found_filename)
                        if resolved_file_id:
                            args["file_id"] = resolved_file_id
                            logger.info(
                                f"✅ Successfully resolved filename '{found_filename}' to file_id: {resolved_file_id}")
                        else:
                            logger.error(f"❌ FAILED to resolve filename '{found_filename}' to file_id")
                            return {
                                "success": False,
                                "function": tool_name,
                                "error": f"Could not resolve filename '{found_filename}' to file_id. File may not exist or be accessible."
                            }

            # At this point, args should ALWAYS have a valid file_id, never a fileName
            if args.get("file_id") and not self._looks_like_file_id(args["file_id"]):
                logger.error(f"❌ CRITICAL ERROR: About to pass fileName as file_id: {args['file_id']}")
                return {
                    "success": False,
                    "function": tool_name,
                    "error": f"Internal error: fileName '{args['file_id']}' was not properly resolved to file_id"
                }

            if "summary_type" not in args and intent.get("summary_type"):
                args["summary_type"] = intent["summary_type"]

            logger.info(f"Delegating summarization to tool_summarize with validated file_id: {args.get('file_id')}")

        # Ensure query_text is provided for tools that need it
        if "query_text" not in args and tool_name in ["search", "search_file", "rag"]:
            args["query_text"] = intent.get("query_text") or ""

        try:
            # Call the tool function
            result = fn(args)
            if result is None:
                result = {"message": "No results found."}

            wrapped_result = {
                "success": True,
                "function": tool_name,
                "result": result
            }

            # Store tool result as a separate conversation entry for context
            self._store_tool_result(user_id, conversation_id, tool_name, result)

            return wrapped_result

        except TypeError as te:
            logger.warning("Tool %s arg mismatch: %s; retrying with minimal args.", tool_name, te)
            try:
                # Retry with minimal args - ENSURE user_id is correct here too
                minimal_args = {
                    "query_text": args.get("query_text", ""),
                    "user_id": user_id  # Use authenticated user_id, not from args
                }

                # For summarize, preserve key parameters in minimal args
                if tool_name == "summarize":
                    # Preserve all potential file identification parameters
                    for key in ["file_id", "filename", "file_name", "fileName", "name"]:
                        if key in args:
                            minimal_args[key] = args[key]

                    # Preserve summary configuration
                    if "summary_type" in args:
                        minimal_args["summary_type"] = args["summary_type"]
                    if "platform" in args:
                        minimal_args["platform"] = args["platform"]

                result = fn(minimal_args)
                wrapped_result = {
                    "success": True,
                    "function": tool_name,
                    "result": result
                }
                self._store_tool_result(user_id, conversation_id, tool_name, result)
                return wrapped_result

            except Exception as e:
                logger.error("Tool %s retry failed: %s", tool_name, e)
                logger.debug(traceback.format_exc())
                return {"success": False, "function": tool_name, "error": str(e)}

        except Exception as e:
            logger.error("Tool %s execution error: %s", tool_name, e)
            logger.debug(traceback.format_exc())
            return {"success": False, "function": tool_name, "error": str(e)}

    def _extract_file_id_from_context(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract file_id (NOT fileName) from recent search results in conversation context.
        """
        try:
            logger.debug("Attempting to extract file_id from conversation context...")

            # Look for the most recent search results
            for i, message in enumerate(reversed(messages)):
                if message.get("role") == "tool" and message.get("name") in ["search", "search_file"]:
                    try:
                        tool_content = json.loads(message.get("content", "{}"))

                        if tool_content.get("success") and tool_content.get("result"):
                            result = tool_content["result"]
                            results = result.get("results", [])

                            if results and len(results) > 0:
                                first_result = results[0]

                                # PRIORITIZE actual file_id fields
                                for field_name in ["file_id", "fileId", "id", "document_id"]:
                                    if field_name in first_result and first_result[field_name]:
                                        potential_file_id = first_result[field_name]
                                        if self._looks_like_file_id(potential_file_id):
                                            logger.info(
                                                f"✅ Found actual file_id in search results: {potential_file_id}")
                                            return potential_file_id

                                # FALLBACK: Return fileName only if no file_id found
                                file_name = first_result.get("fileName")
                                if file_name:
                                    logger.info(f"No file_id found, returning fileName for resolution: {file_name}")
                                    return file_name

                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.debug(f"Could not parse tool content: {e}")
                        continue

            logger.debug("No file reference found in recent conversation context")
            return None

        except Exception as e:
            logger.error(f"Error extracting file reference from context: {e}", exc_info=True)
            return None

    def _extract_filename_from_context(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract filename from recent search results."""
        try:
            for message in reversed(messages):
                if message.get("role") == "tool" and message.get("name") in ["search", "search_file"]:
                    try:
                        tool_content = json.loads(message.get("content", "{}"))
                        if tool_content.get("success") and tool_content.get("result"):
                            result = tool_content["result"]
                            results = result.get("results", [])
                            if results and len(results) > 0:
                                first_result = results[0]
                                filename = first_result.get("fileName")
                                if filename:
                                    return filename
                    except (json.JSONDecodeError, KeyError):
                        continue
            return None
        except Exception as e:
            logger.error(f"Error extracting filename from context: {e}")
            return None

    def _resolve_filename_to_file_id(self, user_id: str, filename: str) -> Optional[str]:
        """
        Resolve a filename to its file_id using search functionality.
        MUST return actual file_id, never fileName.
        """
        try:
            logger.info(f"🔍 Resolving filename '{filename}' to file_id for user {user_id}")

            # Import search function
            try:
                from .search import search_files_by_name
            except ImportError:
                logger.error("Could not import search_files_by_name function")
                return None

            # Try exact match first, then partial match
            search_strategies = [
                {"exact_match": True, "limit": 1},
                {"exact_match": False, "limit": 3}
            ]

            for strategy in search_strategies:
                try:
                    logger.debug(f"Trying search strategy: {strategy}")
                    results = search_files_by_name(
                        file_name=filename,
                        user_id=user_id,
                        **strategy
                    )

                    if results and len(results) > 0:
                        for result in results:
                            # Look for actual file_id in the result
                            for field_name in ["file_id", "fileId", "id", "document_id"]:
                                potential_file_id = result.get(field_name)
                                if potential_file_id and self._looks_like_file_id(potential_file_id):
                                    logger.info(f"✅ Successfully resolved '{filename}' to file_id: {potential_file_id}")
                                    return potential_file_id

                            # Log what we found for debugging
                            logger.debug(f"Search result fields: {list(result.keys())}")

                except Exception as strategy_error:
                    logger.debug(f"Search strategy {strategy} failed: {strategy_error}")
                    continue

            logger.error(f"❌ Could not resolve filename '{filename}' to any valid file_id")
            return None

        except Exception as e:
            logger.error(f"Error resolving filename '{filename}' to file_id: {e}")
            return None

    def _looks_like_file_id(self, value: str) -> bool:
        """
        Check if a string looks like a file_id vs a fileName.
        file_id should NOT have file extensions and should be long/UUID-like.
        """
        if not value or not isinstance(value, str):
            return False

        import re

        # If it has common file extensions, it's definitely a fileName, not file_id
        file_extension_pattern = r'\.(pdf|docx?|xlsx?|pptx?|txt|csv|png|jpg|jpeg)$'
        if re.search(file_extension_pattern, value.lower()):
            logger.debug(f"'{value}' has file extension - treating as fileName")
            return False

        # If it contains spaces or parentheses, likely a fileName
        if ' ' in value or '(' in value or ')' in value:
            logger.debug(f"'{value}' contains spaces/parentheses - treating as fileName")
            return False

        # UUID pattern (with or without hyphens)
        uuid_pattern = r'^[a-fA-F0-9]{8}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{12}$'
        if re.match(uuid_pattern, value):
            logger.debug(f"'{value}' matches UUID pattern - treating as file_id")
            return True

        # Long alphanumeric string without spaces (likely file_id)
        if len(value) > 15 and re.match(r'^[a-zA-Z0-9_\-\.]+$', value) and not re.search(r'[A-Z][a-z]', value):
            logger.debug(f"'{value}' matches long ID pattern - treating as file_id")
            return True

        logger.debug(f"'{value}' does not match file_id patterns - treating as fileName")
        return False

    def _store_tool_result(self, user_id: str, conversation_id: str, tool_name: str, result: Any) -> None:
        """Store tool execution result for context in future conversations."""
        try:
            # Create a summary of tool result for storage
            tool_summary = f"Tool '{tool_name}' executed"
            if isinstance(result, dict):
                if "message" in result:
                    tool_summary += f": {result['message']}"
                elif "summary" in result:
                    tool_summary += f": {result['summary']}"
                else:
                    tool_summary += f" with {len(result)} result items"
            else:
                tool_summary += f": {str(result)[:200]}..."

            # Store as a system message for context using the same conversation_id
            # Generate a unique conversation_id for tool results to avoid confusion
            tool_conversation_id = f"{conversation_id}_tool_{tool_name}_{uuid.uuid4().hex[:8]}"

            self.memory_manager.store_conversation(
                user_id=user_id,
                conversation_id=tool_conversation_id,
                user_query=f"[TOOL_EXECUTION] {tool_name}",
                agent_response=tool_summary,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            logger.error("Failed to store tool result: %s", e)
            # Don't fail the main operation if tool result storage fails

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_json_loads(data: Any) -> Dict[str, Any]:
        """Safely parse JSON data with fallbacks."""
        if isinstance(data, dict):
            return data
        if not data:
            return {}
        try:
            return json.loads(data)
        except Exception:
            try:
                # Attempt to coerce python-style dict string
                cleaned = data.replace("'", '"')
                return json.loads(cleaned)
            except Exception:
                logger.warning("Failed to parse tool args: %s", data)
                return {}


# ---------------------------------------------------------------------------
# Global singleton wiring (optional convenience)
_brain_instance: Optional[ReActBrain] = None


def initialize_brain(
        azure_openai_client: AzureOpenAI,
        chat_deployment: str = "gpt-4o",
        memory_manager: Optional[CosmosMemoryManager] = None,
        max_reasoning_steps: int = 5,
        temperature: float = 0.1,
        conversation_history_limit: int = 10,
) -> None:
    """Initialize the global ReAct brain instance."""
    global _brain_instance
    _brain_instance = ReActBrain(
        azure_openai_client=azure_openai_client,
        chat_deployment=chat_deployment,
        memory_manager=memory_manager,
        max_reasoning_steps=max_reasoning_steps,
        temperature=temperature,
        conversation_history_limit=conversation_history_limit,
    )


def reason_and_act(user_id: str, user_input: str, conversation_id: Optional[str] = None) -> str:
    """Global convenience wrapper."""
    if _brain_instance is None:
        raise RuntimeError("Brain not initialized. Call initialize_brain() first.")
    return _brain_instance.reason_and_act(user_id=user_id, user_input=user_input, conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# Back-compat CLI smoke test
if __name__ == "__main__":  # pragma: no cover
    import os

    # Minimal environment-driven client setup
    _api_key = os.getenv("OPENAI_API_KEY")
    _endpoint = "https://weez-openai-resource.openai.azure.com/"
    _api_version = "2024-12-01-preview"
    client = AzureOpenAI(api_key=_api_key, azure_endpoint=_endpoint, api_version=_api_version)

    initialize_brain(client, chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"))

    print("Type a query (Ctrl+C to quit)...")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            try:
                # Generate a conversation ID for this session
                conv_id = str(uuid.uuid4())
                print(reason_and_act("demo", q, conv_id))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print("Error:", e)
    except KeyboardInterrupt:
        print("\nBye!")


# ---------------------------------------------------------------------------
# Local helper for tool call dict fallback

def _tool_call_to_dict(tc: Any) -> Dict[str, Any]:
    """Convert tool call object to dict format."""
    try:
        return {
            "id": getattr(tc, "id", None),
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": getattr(getattr(tc, "function", None), "name", None),
                "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
            },
        }
    except Exception:
        return {"id": None, "type": "function", "function": {"name": None, "arguments": "{}"}}