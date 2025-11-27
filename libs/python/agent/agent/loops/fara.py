"""
Fara-7B agent loop implementation using liteLLM.

Fara-7B is Microsoft's agentic small language model (7B parameters) based on Qwen 2.5-VL,
designed for computer use tasks with visual perception and direct coordinate prediction.

The model outputs chain-of-thought reasoning followed by tool calls with grounded arguments.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)

from ..decorators import register_agent
from ..loops.base import AsyncAgentConfig
from ..responses import (
    convert_completion_messages_to_responses_items,
    convert_responses_items_to_completion_messages,
    make_reasoning_item,
)
from ..types import AgentCapability

# Fara Computer Use tool schema
FARA_COMPUTER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. "
            "You must click on desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need to wait and "
            "take successive screenshots to see the results of your actions.\n"
            "* Whenever you intend to move the cursor to click on an element, consult a screenshot to "
            "determine the coordinates before moving the cursor.\n"
            "* Make sure to click buttons, links, icons, etc with the cursor tip in the center of the element."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "description": "The action to perform.",
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "scroll",
                        "visit_url",
                        "web_search",
                        "history_back",
                        "pause_and_memorize_fact",
                        "wait",
                        "terminate",
                        "screenshot",
                    ],
                    "type": "string",
                },
                "keys": {
                    "description": "Required only by action=key. Key or key combination to press.",
                    "type": "string",
                },
                "text": {
                    "description": "Required for action=type (text to type) or action=web_search (query).",
                    "type": "string",
                },
                "coordinate": {
                    "description": "(x, y): Pixel coordinates from top-left. Required for click/move/scroll actions.",
                    "type": "array",
                    "items": {"type": ["number", "integer"]},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "url": {
                    "description": "URL to visit (action=visit_url).",
                    "type": "string",
                },
                "fact": {
                    "description": "Fact to memorize (action=pause_and_memorize_fact).",
                    "type": "string",
                },
                "scroll_amount": {
                    "description": "Scroll distance in pixels. Positive=down, negative=up.",
                    "type": "number",
                },
                "status": {
                    "description": "Task status (action=terminate).",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
        },
    },
}


def _parse_fara_output(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Parse Fara model output to extract thought and action.

    Fara outputs follow a pattern like:
    Thought #N: [reasoning about next step]
    Action #N: executing tool '[action_type]' with arguments {...}

    Or potentially:
    <tool_call>{"name": "computer_use", "arguments": {...}}</tool_call>

    Returns:
        Tuple of (thought_text, action_dict)
    """
    thought = None
    action = None

    # Try to extract thought
    thought_match = re.search(r"Thought\s*#?\d*:\s*(.+?)(?=\n(?:Action|$))", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Try to extract action - format 1: executing tool 'action_type' with arguments {...}
    action_match = re.search(
        r"Action\s*#?\d*:\s*executing tool\s+'([^']+)'\s+with arguments\s+(\{[^}]*\})",
        text,
        re.DOTALL,
    )
    if action_match:
        action_name = action_match.group(1)
        try:
            action_args = json.loads(action_match.group(2))
            action = {"name": action_name, "arguments": action_args}
        except json.JSONDecodeError:
            pass

    # Try to extract action - format 2: <tool_call>...</tool_call>
    if not action:
        tool_call_match = re.search(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text)
        if tool_call_match:
            try:
                action = json.loads(tool_call_match.group(1))
            except json.JSONDecodeError:
                pass

    # Try to extract action - format 3: direct JSON tool call
    if not action:
        json_match = re.search(
            r'\{[^}]*"(?:name|action)"[^}]*:[^}]*"(?:computer_use|[a-z_]+)"[^}]*\}',
            text,
            re.DOTALL,
        )
        if json_match:
            try:
                action = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    return thought, action


def convert_fara_action_to_computer_action(action_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert Fara action format to CUA Computer Calls action schema.

    Fara format (example):
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}

    CUA format (example):
        {"action": "left_click", "x": x, "y": y}
    """
    if not isinstance(action_dict, dict):
        return None

    # Extract arguments
    args = action_dict.get("arguments", action_dict)
    if not isinstance(args, dict):
        return None

    action = args.get("action")
    if not isinstance(action, str):
        return None

    # Extract coordinates
    coord = args.get("coordinate")
    x = y = None
    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
        try:
            x = int(round(float(coord[0])))
            y = int(round(float(coord[1])))
        except Exception:
            x = y = None

    # Map actions
    action_lower = action.lower()

    if action_lower == "left_click":
        if x is None or y is None:
            return None
        return {"action": "click", "x": x, "y": y}

    elif action_lower == "mouse_move":
        if x is None or y is None:
            return None
        return {"action": "move", "x": x, "y": y}

    elif action_lower == "key":
        keys = args.get("keys", "")
        if not keys:
            return None
        # Convert key string to list (split by '+' for combinations)
        key_list = [k.strip() for k in keys.split("+")]
        return {"action": "keypress", "keys": key_list}

    elif action_lower == "type":
        text = args.get("text", "")
        if not text:
            return None
        return {"action": "type", "text": text}

    elif action_lower == "scroll":
        if x is None or y is None:
            x = y = 0
        scroll_amount = args.get("scroll_amount", 0)
        # Positive = down, negative = up
        return {"action": "scroll", "x": x, "y": y, "scroll_x": 0, "scroll_y": int(scroll_amount)}

    elif action_lower == "wait":
        return {"action": "wait"}

    elif action_lower == "screenshot":
        return {"action": "screenshot"}

    elif action_lower == "visit_url":
        # Not a direct UI action, could be handled separately
        # For now, return None and let the text output guide the user
        return None

    elif action_lower == "web_search":
        # Not a direct UI action
        return None

    elif action_lower == "history_back":
        # Could map to a key press (Alt+Left or Backspace)
        return {"action": "keypress", "keys": ["alt", "Left"]}

    elif action_lower == "pause_and_memorize_fact":
        # Not a direct UI action
        return None

    elif action_lower == "terminate":
        # Task completion - not a UI action
        return None

    return None


@register_agent(models=r"(?i).*fara.*", priority=0)
class FaraConfig(AsyncAgentConfig):
    """Fara-7B agent configuration using liteLLM."""

    async def predict_step(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: Optional[int] = None,
        stream: bool = False,
        computer_handler=None,
        use_prompt_caching: Optional[bool] = False,
        _on_api_start=None,
        _on_api_end=None,
        _on_usage=None,
        _on_screenshot=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Predict the next step using Fara-7B model.

        Args:
            messages: Input messages following Responses format
            model: Model name to use (e.g., "microsoft/Fara-7B")
            tools: Optional list of tool schemas (defaults to FARA_COMPUTER_TOOL)
            max_retries: Maximum number of retries for API calls
            stream: Whether to stream the response
            computer_handler: Computer handler for taking screenshots
            use_prompt_caching: Whether to use prompt caching
            _on_api_start: Callback for API start
            _on_api_end: Callback for API end
            _on_usage: Callback for usage tracking
            _on_screenshot: Callback for screenshot events

        Returns:
            Dict with "output" and "usage" keys
        """
        # Convert Responses items to completion messages
        completion_messages = convert_responses_items_to_completion_messages(messages)

        # Use Fara computer tool if no tools specified
        if not tools:
            tools = [FARA_COMPUTER_TOOL]

        # Prepare system message with computer use instructions
        system_message = {
            "role": "system",
            "content": (
                "You are a computer use agent. Your task is to help users accomplish tasks "
                "by interacting with a computer using mouse and keyboard actions.\n"
                "For each step, provide your reasoning (Thought) and then the action to take (Action).\n"
                "Always stop and ask for permission at Critical Points: situations requiring user "
                "permission or sensitive information (checkout, purchases, calls, emails, etc.)."
            ),
        }

        # Insert system message if not present
        if not completion_messages or completion_messages[0].get("role") != "system":
            completion_messages.insert(0, system_message)

        # Prepare API call kwargs
        api_kwargs = {
            "model": model,
            "messages": completion_messages,
            "tools": tools,
            "stream": stream,
        }

        # Add any additional kwargs
        if kwargs:
            api_kwargs.update(kwargs)

        # Callbacks
        if _on_api_start:
            await _on_api_start(api_kwargs)

        # Call liteLLM
        response = await litellm.acompletion(**api_kwargs)

        if _on_api_end:
            await _on_api_end(api_kwargs, response)

        # Parse response
        if not response.choices or not response.choices[0].message:
            return {"output": [], "usage": {}}

        message = response.choices[0].message
        response_items = []

        # Check for reasoning content
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            reasoning_item = make_reasoning_item(reasoning_content).model_dump()
            response_items.append(reasoning_item)

        # Check for tool calls (standard OpenAI format)
        if hasattr(message, "tool_calls") and message.tool_calls:
            # Use standard conversion for OpenAI-style tool calls
            items = convert_completion_messages_to_responses_items([message.model_dump()])
            response_items.extend(items)
        else:
            # Parse text content for Fara-specific format
            content = message.content or ""
            thought, action = _parse_fara_output(content)

            # Add thought as reasoning or text
            if thought:
                reasoning_item = make_reasoning_item(thought).model_dump()
                response_items.append(reasoning_item)

            # Convert action to computer call
            if action:
                computer_action = convert_fara_action_to_computer_action(action)
                if computer_action:
                    # Create a computer call item
                    import uuid
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    computer_call = {
                        "type": "computer_call",
                        "call_id": call_id,
                        "action": computer_action,
                        "status": "completed",
                    }
                    response_items.append(computer_call)

            # If we have remaining text that's not thought/action, add as output text
            if content and not (thought and action):
                # Clean up the content
                clean_content = content
                if thought:
                    clean_content = clean_content.replace(thought, "")
                clean_content = re.sub(r"Thought\s*#?\d*:", "", clean_content)
                clean_content = re.sub(r"Action\s*#?\d*:.*", "", clean_content, flags=re.DOTALL)
                clean_content = clean_content.strip()

                if clean_content:
                    from ..responses import make_output_text_item
                    text_item = make_output_text_item(clean_content).model_dump()
                    response_items.append(text_item)

        # Extract usage information
        response_usage = {}
        if hasattr(response, "usage") and response.usage:
            response_usage = {
                **LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
                    response.usage
                ).model_dump(),
                "response_cost": getattr(response, "_hidden_params", {}).get("response_cost", 0.0),
            }
            if _on_usage:
                await _on_usage(response_usage)

        return {"output": response_items, "usage": response_usage}

    async def predict_click(
        self, model: str, image_b64: str, instruction: str, **kwargs
    ) -> Optional[Tuple[int, int]]:
        """
        Predict click coordinates using Fara-7B model.

        Args:
            model: Model name to use
            image_b64: Base64 encoded image
            instruction: Instruction for where to click

        Returns:
            Tuple with (x, y) coordinates or None
        """
        try:
            # Create a simple click instruction prompt
            click_prompt = f"""Look at the screenshot and identify where to click for: {instruction}

Respond with:
Thought: [your reasoning]
Action: executing tool 'computer_use' with arguments {{"action": "left_click", "coordinate": [x, y]}}

Where x, y are pixel coordinates from the top-left corner."""

            # Prepare messages for liteLLM
            litellm_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": click_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ]

            # Prepare API call kwargs
            api_kwargs = {
                "model": model,
                "messages": litellm_messages,
            }
            api_kwargs.update(kwargs)

            # Call liteLLM
            response = await litellm.acompletion(**api_kwargs)

            # Extract response content
            if not response.choices or not response.choices[0].message:
                return None

            response_content = response.choices[0].message.content or ""

            # Parse the response
            _, action = _parse_fara_output(response_content)

            if action:
                computer_action = convert_fara_action_to_computer_action(action)
                if computer_action and computer_action.get("action") == "click":
                    x = computer_action.get("x")
                    y = computer_action.get("y")
                    if x is not None and y is not None:
                        return (x, y)

            return None

        except Exception as e:
            # Log error and return None
            print(f"Error in predict_click: {e}")
            return None

    def get_capabilities(self) -> List[AgentCapability]:
        """
        Get list of capabilities supported by this agent config.

        Returns:
            List of capability strings
        """
        return ["step", "click"]
