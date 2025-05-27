#!/usr/bin/env python3
"""
Integrates with signal-cli-rest-api and LLM services (llama.cpp/Ollama/Others). Should work with Openai-api-compatible endpoints.
"""

import asyncio
import json
import base64
import traceback
import os
from typing import Dict, Any, List, Optional, Callable, Awaitable
import importlib
import logging
import aiohttp
import aiofiles
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {}
TYPING_TASKS = {}
COMMANDS = {}


def load_config(config_path: str = "./files/config/config.json") -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.critical(f"Failed to load config: {e}")
        raise


async def http_request(method: str, url: str, **kwargs) -> Optional[Any]:
    try:
        async with aiohttp.ClientSession() as session:
            async with getattr(session, method)(url, **kwargs) as resp:
                if resp.status in [200, 201, 202, 204]:
                    if method == 'get':
                        return await resp.read()
                    elif resp.content_type == 'application/json':
                        return await resp.json()
                    return {'status': resp.status, 'text': await resp.text()}
                logger.error(f"HTTP {method.upper()} error: {resp.status} - {await resp.text()}")
                return None
    except Exception as e:
        logger.error(f"HTTP {method} error: {e}")
        return None


async def post_json(url: str, data: Dict, headers: Dict = None) -> Optional[Dict]:
    return await http_request('post', url, json=data, headers=headers)


async def post_form(url: str, data) -> Optional[Dict]:
    return await http_request('post', url, data=data)


async def get_binary(url: str) -> Optional[bytes]:
    return await http_request('get', url)


async def put_json(url: str, data: Dict) -> bool:
    result = await http_request('put', url, json=data)
    return result is not None


async def delete_json(url: str, data: Dict = None) -> bool:
    kwargs = {'json': data} if data else {}
    result = await http_request('delete', url, **kwargs)
    return result is not None

def get_safe_recipient_string(recipient: str) -> str:
    # Sanitize recipient for filesystem names
    return "".join(c for c in recipient if c.isalnum() or c in "+-_").strip()

def get_recipient_memory_file(recipient: str) -> str:
    return f"files/memory/{get_safe_recipient_string(recipient)}_memory.json"


async def load_memory(file_path: str) -> List[Dict]:
    try:
        async with aiofiles.open(file_path, "r") as f:
            fread = await f.read()
            return json.loads(fread)["messages"]
    except FileNotFoundError:
        logger.info(f"Memory file not found: {file_path}.")
        logger.info(f"Creating: {file_path}.")
        if CONFIG.get("llm_model_options", {}).get("system_prompt"):
            return [{"system": CONFIG["llm_model_options"]["system_prompt"]}]
        else:
            return []
    except Exception as e:
        logger.error(f"Error loading memory: {e}")
        return []


async def save_memory(recipient: str, memory: List[Dict]) -> None:
    if CONFIG.get("save_memory"):
        file_path = get_recipient_memory_file(recipient)
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps({"messages": memory}))
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")


def get_memory_for_recipient(recipient: str) -> List[Dict]:
    recipient_file = get_recipient_memory_file(recipient)
    memory = load_memory(recipient_file)
   
    # Set system prompt from config if no memory and system prompt configured
    if not memory and CONFIG.get("llm_model_options", {}).get("system_prompt"):
        memory = [{"system": CONFIG["llm_model_options"]["system_prompt"]}]

    return memory

async def reset_memory(recipient: str) -> None:
    memory = await load_memory(get_recipient_memory_file(recipient))
    if memory and "system" in memory[0]:
        memory = [memory[0]]  # Keep system prompt
    else:
        memory = []
    await save_memory(recipient, memory)

async def set_system_prompt(recipient: str, prompt: str) -> None:
    memory = await load_memory(get_recipient_memory_file(recipient))
    if memory and "system" in memory[0]:
        memory[0]["system"] = prompt
    
    await save_memory(recipient, memory)

# LLM Provider Base Class
class LLMProvider:
    def __init__(self, service_url: str, model_options: Dict[str, Any]):
        self.service_url = service_url
        self.model_options = model_options
        self.endpoint = f"{service_url}/v1/chat/completions"
    
    def prepare_payload(self, memory: List[Dict], attachments: List[Dict] = None) -> Dict:
        raise NotImplementedError
    
    def prepare_headers(self, api_key: str = None) -> Dict[str, str]:
        raise NotImplementedError
    
    def handle_attachments(self, attachments: List[Dict]) -> List[Dict]:
        raise NotImplementedError


def get_llm_provider(provider_name: str, service_url: str, model_options: Dict[str, Any]) -> Optional[LLMProvider]:
    try:
        module_import = importlib.import_module(f"files.provider.{provider_name}")
        module = getattr(module_import, f"{provider_name.capitalize()}")
        return module(service_url, model_options)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load LLM provider '{provider_name}': {e}")
        return None


def prepare_llm_payload(memory: List[Dict], attachments: List[Dict] = None) -> Dict:
    return LLM_PROVIDER.prepare_payload(memory, attachments)


def prepare_headers(api_key: str = None) -> Dict[str, str]:
    return LLM_PROVIDER.prepare_headers(api_key)


def prepare_llm_payload(memory: List[Dict], attachments: List[Dict] = None) -> Dict:
    return LLM_PROVIDER.prepare_payload(memory, attachments)


def prepare_headers(api_key: str = None) -> Dict[str, str]:
    return LLM_PROVIDER.prepare_headers(api_key)


async def call_llm(recipient: str, text: str, attachments: List[Dict] = None) -> Optional[str]:
    try:
        # Add user message to memory
        memory = await load_memory(get_recipient_memory_file(recipient))
        memory.append({"user": text.rstrip() if text else ""})
        
        # Filter attachments through provider.
        filtered_attachments = LLM_PROVIDER.handle_attachments(attachments) if attachments else None
        
        payload = prepare_llm_payload(memory, filtered_attachments)
        headers = prepare_headers(CONFIG.get("llm_api_key"))
        
        response = await post_json(LLM_PROVIDER.endpoint, payload, headers)
        if not response:
            return "Failed to get LLM response"
        
        content = response["choices"][0]["message"]["content"]
        memory.append({"assistant": content.rstrip()})
        
        # Save interaction
        await save_memory(recipient, memory)
        
        return content
        
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


async def download_attachment(attachment_id: str) -> Optional[str]:
    try:
        url = f"http://{CONFIG['signal_service']}/v1/attachments/{attachment_id}"
        content = await get_binary(url)
        return base64.b64encode(content).decode("utf-8") if content else None
    except Exception as e:
        logger.error(f"Error downloading attachment: {e}")
        return None


async def upload_attachment(attachment: Dict) -> Optional[str]:
    try:
        binary_data = base64.b64decode(attachment["data"])
        form_data = aiohttp.FormData()
        form_data.add_field(
            "attachment",
            binary_data,
            filename=attachment.get("filename", "image.jpg"),
            content_type=attachment.get("content_type", "image/jpeg")
        )
        
        url = f"http://{CONFIG['signal_service']}/v1/attachments"
        response = await post_form(url, form_data)
        return response.get("id") if response else None
    except Exception as e:
        logger.error(f"Error uploading attachment: {e}")
        return None


async def save_attachment(recipient: str, attachment_id: str, data: str, filename: str = None) -> None:
    if CONFIG.get("save_attachments"):
        try:
            recipient_folder = f"files/attachments/{get_safe_recipient_string(recipient)}"
            os.makedirs(f"{recipient_folder}", exist_ok=True)
            filepath = f"{recipient_folder}/{attachment_id}"
            binary_data = base64.b64decode(data)
            async with aiofiles.open(filepath, "wb") as f:
                await f.write(binary_data)
        except Exception as e:
            logger.error(f"Failed to save attachment: {e}")


async def send_typing_indicator(recipient: str, action: str) -> None:
    url = f"http://{CONFIG['signal_service']}/v1/typing-indicator/{CONFIG['phone_number']}"
    payload = {"recipient": recipient}
    
    if action == "start":
        await put_json(url, payload)
    elif action == "stop":
        await delete_json(url, payload)

async def maintain_typing(recipient: str) -> None:
    try:
        while True:
            await send_typing_indicator(recipient, "start")
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        await send_typing_indicator(recipient, "stop")
        raise


async def start_typing(recipient: str) -> None:
    await stop_typing(recipient)  # Stop any existing
    TYPING_TASKS[recipient] = asyncio.create_task(maintain_typing(recipient))


async def stop_typing(recipient: str) -> None:
    if recipient in TYPING_TASKS:
        TYPING_TASKS[recipient].cancel()
        try:
            await TYPING_TASKS[recipient]
        except asyncio.CancelledError:
            pass
        del TYPING_TASKS[recipient]


def register_command(command: str, handler: Callable[[], Awaitable[None]]) -> None:
    COMMANDS[command] = handler
    logger.info(f"Registered command: {command}")

# TODO: Moore error handling.
async def handle_command(text: str, recipient: str = None) -> bool:
    if text:
        cmd = text.split()
        if cmd[0] in COMMANDS:
            try:
                # Pass recipient and, if applicable, command parameter(s) to command if it accepts it
                import inspect
                sig = inspect.signature(COMMANDS[cmd[0]])
                if {"recipient", "prompt"}.issubset(sig.parameters):
                    if len(cmd) > 1:
                        await COMMANDS[cmd[0]](recipient, " ".join(cmd[1:]))
                elif 'recipient' in sig.parameters:
                    await COMMANDS[cmd[0]](recipient)
                else:
                    await COMMANDS[cmd[0]]()
                return True
            except Exception as e:
                logger.error(f"Command error: {e}")
    return False


async def reset_memory_command(recipient: str = None) -> None:
    if recipient:
        await reset_memory(recipient)
        logger.info(f"Memory reset for recipient {recipient} via command")
    else:
        logger.warning("Reset command called without recipient")

async def set_system_prompt_command(recipient: str, prompt: str) -> None:
    if recipient:
        await set_system_prompt(recipient, prompt)
        logger.info(f"System prompt changed for recipient {recipient} via command")
    else:
        logger.warning("System prompt command called without recipient")


# Signal Message Functions
def parse_signal_message(raw_message: str) -> Optional[Dict]:
    try:
        data = json.loads(raw_message)
        envelope = data.get("envelope", {})
        
        # Get message type and get data
        message_data = None
        if "dataMessage" in envelope:
            message_data = envelope["dataMessage"]
        elif "syncMessage" in envelope and "sentMessage" in envelope["syncMessage"]:
            message_data = envelope["syncMessage"]["sentMessage"]
        else:
            return None

        # Build result
        result = {
            "source": envelope.get("source"),
            "timestamp": envelope.get("timestamp"),
            "text": message_data.get("message", ""),
            "attachments": []
        }
        # Handle attachments
        if "attachments" in message_data and message_data["attachments"]:
            result["attachments"] = message_data.get("attachments", [])
        
        # Get recipient
        if "groupInfo" in message_data:
            result["recipient"] = message_data["groupInfo"]["groupId"]
        else:
            result["recipient"] = envelope.get("source")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to parse message: {e}")
        return None


async def process_attachments(recipient: str, attachments_data: List[Dict]) -> List[Dict]:
    processed = []
    for att in attachments_data:
        att_id = att.get("id")
        if att_id:
            data = await download_attachment(att_id)
            if data:
                processed.append({
                    "id": att_id,
                    "content_type": att.get("contentType", ""),
                    "filename": att.get("filename", ""),
                    "data": data
                })
                await save_attachment(recipient, att_id, data, att.get("filename"))
    return processed


async def send_signal_message(recipient: str, text: str, attachments: List[Dict] = None) -> None:
    url = f"http://{CONFIG['signal_service']}/v2/send"
    payload = {
        "message": text,
        "number": CONFIG["phone_number"],
        "recipients": [recipient]
    }
    
    if attachments:
        attachment_ids = []
        for att in attachments:
            att_id = await upload_attachment(att)
            if att_id:
                attachment_ids.append(att_id)
        if attachment_ids:
            payload["attachments"] = attachment_ids
    
    await post_json(url, payload)


async def handle_signal_message(raw_message: str) -> None:
    try:
        message = parse_signal_message(raw_message)
        if not message or (not message.get("text") and not message.get("attachments")):
            return
          
        text = message.get("text", "")
        recipient = message.get("recipient")
        
        # Check for commands first
        if await handle_command(text, recipient):
            return
        
        # Start typing indicator
        await start_typing(recipient)
        
        try:
            # Process attachments if any
            attachments = []
            if "attachments" in message:
                attachments = await process_attachments(recipient, message["attachments"])
            
            # Get LLM response
            response = await call_llm(recipient, text, attachments)
            
            # Stop typing and send response
            await stop_typing(recipient)
            if response:
                await send_signal_message(recipient, response)
                
        except Exception as e:
            await stop_typing(recipient)
            logger.error(f"Error processing message: {e}")
            
    except Exception as e:
        logger.error(f"Error in message handling: {e}")


# Signal loop
async def signal_loop(uri: str) -> None:
    retry_count = 0
    max_retries = 5
    
    while True:
        try:
            async with websockets.connect(uri, ping_interval=None) as websocket:
                logger.info(f"Connected to WebSocket at {uri}")
                retry_count = 0
                
                async for message in websocket:
                    await handle_signal_message(message)
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
            retry_count += 1
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            retry_count += 1
        
        # Exponential backoff
        wait_time = min(30, 2 ** retry_count)
        logger.info(f"Reconnecting in {wait_time}s... (Attempt {retry_count}/{max_retries})")
        await asyncio.sleep(wait_time)
        
        if retry_count >= max_retries:
            logger.error(f"Max retries exceeded. Waiting 60s...")
            retry_count = 0
            await asyncio.sleep(60)


async def main(llm_api_key: str = None):
    global CONFIG, LLM_PROVIDER
    
    try:
        # Load configuration
        CONFIG = load_config()
        if llm_api_key:
            CONFIG["llm_api_key"] = llm_api_key
        # Initialize LLM provider
        LLM_PROVIDER = get_llm_provider(
            CONFIG["llm_service_provider"],
            CONFIG["llm_service_url"], 
            CONFIG["llm_model_options"]
        )
        if not LLM_PROVIDER:
            raise ValueError(f"Failed to initialize LLM provider: {CONFIG['llm_service_provider']}")
        
        # Ensure directories exist
        os.makedirs("files/memory", exist_ok=True)
        os.makedirs("files/attachments", exist_ok=True)
        
        # Register reset command
        register_command("/reset", reset_memory_command)
        register_command("/prompt", set_system_prompt_command)
        
        # Start WebSocket listener
        ws_uri = f"ws://{CONFIG['signal_service']}/v1/receive/{CONFIG['phone_number']}"
        logger.info("Starting Signal API Relay service...")
        await signal_loop(ws_uri)
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        # Load API key from environment
        api_key = os.getenv("API_KEY", "")
        
        asyncio.run(main(api_key))
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        exit(1)
