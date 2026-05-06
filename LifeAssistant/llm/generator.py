# llm/generator.py
# ============================================================
# LLM GENERATOR — Uses Groq API (FREE tier)
#
# Free limits: ~14,400 requests/day on llama-3.3-70b
# Extremely fast inference (runs on custom AI chips)
# No credit card required.
#
# Get key: https://console.groq.com → API Keys
# ============================================================

# llm/generator.py
# Groq API with streaming support

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from typing import Iterator
from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS


SYSTEM_PROMPT = """You are a personal life assistant with access to the user's private data including their notes, PDFs, schedules, and past conversations.

Your role:
1. Answer questions using the provided context from their personal data
2. Help them manage their studies, tasks, and daily routines
3. Provide decision support when asked
4. Remember details they've shared and reference them naturally
5. Be concise, warm, and genuinely helpful

Rules:
- ONLY use information from the provided context or the conversation history
- If the context doesn't contain relevant information, say so honestly
- Never make up facts about the user's schedule, notes, or data
- Cite which document/source you're drawing from when relevant"""


class LLMGenerator:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set!\n"
                "Windows: $env:GROQ_API_KEY='gsk_...'\n"
                "Mac/Linux: export GROQ_API_KEY='gsk_...'"
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        print(f"🤖 LLM ready: {LLM_MODEL} (Groq - Free, Streaming enabled)")

    def _build_messages(self, query: str, context: str,
                        username: str, conversation_history: list) -> list:
        """Build the messages array for the API call."""
        if context:
            user_content = f"""Here is relevant context from {username}'s personal data:

{context}

---

Based on this context, please answer:
{query}"""
        else:
            user_content = query

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in (conversation_history or []):
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(self, query: str, context: str = "",
                 username: str = "user",
                 conversation_history: list = None) -> str:
        """Non-streaming generation (kept for compatibility)."""
        messages = self._build_messages(query, context, username, conversation_history)
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def generate_stream(self, query: str, context: str = "",
                        username: str = "user",
                        conversation_history: list = None) -> Iterator[str]:
        """
        Streaming generation — yields text chunks as they arrive.
        Each chunk is a small piece of the response (a few words).
        """
        messages = self._build_messages(query, context, username, conversation_history)

        # stream=True tells Groq to send tokens as they're generated
        stream = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7,
            stream=True  # ← the key change
        )

        for chunk in stream:
            # Each chunk has a delta with the new text
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def generate_decision_support(self, situation: str,
                                   context: str, username: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Context from {username}'s data:
{context}

Situation: {situation}

Structure your response as:
1. **Situation Analysis**
2. **Recommendations**
3. **Priority**
4. **Potential Conflicts**"""}
        ]
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7,
        )
        return response.choices[0].message.content