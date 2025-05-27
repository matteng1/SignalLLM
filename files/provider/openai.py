from typing import Dict, Any, List
import sys
import os

class Openai:
   
    def __init__(self, service_url: str, model_options: Dict[str, Any]):
        self.service_url = service_url
        self.model_options = model_options
        self.endpoint = f"{service_url}/v1/chat/completions"
    
    def prepare_payload(self, memory: List[Dict], attachments: List[Dict] = None) -> Dict:
        messages = []
        for i, msg in enumerate(memory):
            for role, content in msg.items():
                if role == "user" and attachments and i == len(memory) - 1:
                    # Multimodal handling
                    message_obj = {"role": role, "content": [{"type": "text", "text": content}]}
                    for att in self.handle_attachments(attachments):
                        message_obj["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{att.get('content_type', 'image/jpeg')};base64,{att.get('data', '')}"}
                        })
                    messages.append(message_obj)
                else:
                    messages.append({"role": role, "content": content})
        
        return {
            "model": self.model_options.get("model", ""),
            "messages": messages,
            "stream": False,
            "keep_alive": self.model_options.get("keep_alive", 5)
        }
    
    def prepare_headers(self, api_key: str = None) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'no-key'}"
        }
    # Filter for images for now.
    def handle_attachments(self, attachments: List[Dict]) -> List[Dict]:
        return [
            att for att in attachments 
            if att.get("content_type", "").startswith("image/") and att.get("content_type") != "image/gif"
        ]
