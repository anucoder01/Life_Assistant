"""LLM generation with ordered provider failover and streaming support."""

import os
import sys
from typing import Iterator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL,
    LLM_MAX_TOKENS, LLM_PROVIDERS,
)


SYSTEM_PROMPT = """You are a personal life assistant with access to the user's private data including their notes, PDFs, schedules, and past conversations.

Your role:
1. Answer questions using the provided context from their personal data
2. Help them manage their studies, tasks, and daily routines
3. Provide decision support when asked
4. Remember details they've shared and reference them naturally
5. Be concise, warm, and helpful

Rules:
- ONLY use information from the provided context or the conversation history
- If the context doesn't contain relevant information, say so honestly
- Never make up facts about the user's schedule, notes, or data
- Cite which document/source you're drawing from when relevant"""


class LLMGenerator:
    """Try each configured provider in order when a request fails."""

    def __init__(self):
        self.providers = []
        for name in LLM_PROVIDERS:
            if name == "groq" and GROQ_API_KEY:
                from groq import Groq
                self.providers.append((name, Groq(api_key=GROQ_API_KEY)))
            elif name == "gemini" and GEMINI_API_KEY:
                from google import genai
                self.providers.append((name, genai.Client(api_key=GEMINI_API_KEY)))

        if not self.providers:
            raise ValueError(
                "No configured LLM provider. Set GROQ_API_KEY and/or "
                "GEMINI_API_KEY, and optionally set LLM_PROVIDERS=groq,gemini."
            )
        print("LLM providers ready: " + ", ".join(p[0] for p in self.providers))

    def _build_messages(self, query: str, context: str, username: str,
                        conversation_history: list) -> list:
        if context:
            user_content = (
                f"Here is relevant context from {username}'s personal data:\n\n"
                f"{context}\n\nBased on this context, please answer:\n{query}"
            )
        else:
            user_content = query
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in (conversation_history or []):
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate_with(self, name: str, client, messages: list) -> str:
        if name == "groq":
            response = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=LLM_MAX_TOKENS, temperature=0.7,
            )
            return response.choices[0].message.content or ""
        prompt = "\n\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in messages if m["role"] != "system"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": LLM_MAX_TOKENS,
                "temperature": 0.7,
            },
        )
        return response.text or ""

    def generate(self, query: str, context: str = "", username: str = "user",
                 conversation_history: list = None) -> str:
        messages = self._build_messages(query, context, username, conversation_history)
        errors = []
        for name, client in self.providers:
            try:
                answer = self._generate_with(name, client, messages)
                if not answer.strip():
                    raise RuntimeError("provider returned an empty response")
                return answer
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"LLM provider {name} failed; trying next provider.")
        raise RuntimeError("All LLM providers failed. " + " | ".join(errors))

    def _stream_with(self, name: str, client, messages: list) -> Iterator[str]:
        if name == "groq":
            stream = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=LLM_MAX_TOKENS, temperature=0.7, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            return

        prompt = "\n\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in messages if m["role"] != "system"
        )
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": LLM_MAX_TOKENS,
                "temperature": 0.7,
            },
        )
        for chunk in stream:
            if getattr(chunk, "text", None):
                yield chunk.text

    def generate_stream(self, query: str, context: str = "", username: str = "user",
                        conversation_history: list = None) -> Iterator[str]:
        """Fail over until a provider emits its first token; after that, errors surface."""
        messages = self._build_messages(query, context, username, conversation_history)
        errors = []
        for name, client in self.providers:
            emitted = False
            try:
                for text in self._stream_with(name, client, messages):
                    if text:
                        emitted = True
                        yield text
                if emitted:
                    return
                raise RuntimeError("provider returned an empty response")
            except Exception as exc:
                if emitted:
                    raise RuntimeError(
                        f"LLM stream from {name} failed after output began: {exc}"
                    ) from exc
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"LLM provider {name} failed before streaming; trying next provider.")
        raise RuntimeError("All LLM providers failed. " + " | ".join(errors))

    def generate_decision_support(self, situation: str, context: str,
                                  username: str) -> str:
        query = (
            f"Context from {username}'s data:\n{context}\n\nSituation: {situation}\n\n"
            "Structure your response as: Situation Analysis; Recommendations; "
            "Priority; Potential Conflicts."
        )
        return self.generate(query, username=username)
