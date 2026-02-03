import inspect
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class ConversationMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class Conversation:
    def __init__(self, system_prompt: Optional[str] = None):
        self.messages: List[ConversationMessage] = []
        self.system_prompt = system_prompt or "You are a helpful assistant."

    def set_system_message(self, message: str) -> None:
        self.system_prompt = message
        
    def add_message(self, role: str, content: str) -> None:
        self.messages.append(ConversationMessage(role, content))
        
    def get_messages(self, llm, include_history: bool = True) -> List[Dict[str, str]]:
        return self.to_list(include_history=include_history, llm=llm)

    def get_last_message(self) -> Optional[Dict[str, str]]:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.to_dict()
        return None

    def to_list(self, include_history: bool = True, llm: Optional[Any] = None) -> List[Dict[str, str]]:
        messages = []
        if include_history:
            chat_messages = [m.to_dict() for m in self.messages]
        else:
            last_user_msg = self.get_last_message()
            chat_messages = [last_user_msg] if last_user_msg else []

        if llm is None or llm.check_role_support("system"):
            messages.append({"role": "system", "content": self.system_prompt})
            messages += chat_messages
        else:
            if chat_messages:
                chat_messages[0]["content"] = f"{self.system_prompt}\n{chat_messages[0]['content']}"
            messages += chat_messages
        return messages

    def to_prompt(self, llm, include_history: bool = True) -> Any:
        messages = self.to_list(include_history=include_history, llm=llm)

        if llm.tokenizer and llm.tokenizer.chat_template is None:
            return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

        if llm.tokenizer and hasattr(llm.tokenizer, "apply_chat_template"):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True
            }

            if "enable_thinking" in inspect.signature(llm.tokenizer.apply_chat_template).parameters:
                kwargs["enable_thinking"] = True
            # kwargs["no_thinking"] = True

            try:
                return llm.tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as e:
                print(f"[Tokenizer Error] Fallback to plain format: {e}")
                return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

        return messages
