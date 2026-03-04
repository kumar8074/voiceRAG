# ===================================================================================
# Project: VoiceRAG
# File: src/services/rag/prompt_builder.py
# Description: Builds prompts for RAG responses
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from typing import List, Dict

class PromptBuilder:

    @staticmethod
    def build_prompt(
        query: str,
        contexts: List[Dict],
        history: List[Dict]
    ) -> str:

        context_text = "\n\n".join([doc["content"] for doc in contexts])

        history_text = ""
        for turn in history[-5:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

        prompt = f"""
You are a helpful assistant answering questions based on provided documents.

Important Rules:
- Respond in the SAME language as the user's question.
- Use the provided document context.
- If the answer is not in the documents, say you don't know.
- Be concise.

Conversation History:
{history_text}

Context Documents:
{context_text}

User Question:
{query}

Answer:
"""
        return prompt