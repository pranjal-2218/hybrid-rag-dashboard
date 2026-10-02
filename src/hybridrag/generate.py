import os
import logging
from typing import List, Dict

logger = logging.getLogger("hybrid-rag-generator")

class Generator:
    def __init__(self):
        """
        Since this app is deployed on Render (which has strict memory limits),
        we leverage external LLM APIs (like Gemini or Groq) for the generation layer.
        This provides state-of-the-art response impact without OOM crashes.
        """
        # Ensure your external API key is set in your Render environment variables!
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.provider = "gemini" if self.api_key else "dummy"
        
        if self.provider == "gemini":
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                logger.warning("google-genai library not found. Falling back to dummy generator.")
                self.provider = "dummy"
                
    def format_prompt(self, query: str, context_chunks: List[dict]) -> str:
        context_str = "\n\n".join(
            [f"Source [{i+1}] (Doc: {c['document_id']}):\n{c['text']}" for i, c in enumerate(context_chunks)]
        )
        prompt = f"""You are an advanced RAG assistant. Answer the user's query based ONLY on the provided context below.
If the context does not contain the answer, politely state that you cannot answer based on the provided documents.
Always cite your sources using the [Number] format.

Context:
{context_str}

Query: {query}
Answer:"""
        return prompt

    def generate_answer(self, query: str, context_chunks: List[dict]) -> str:
        if not context_chunks:
            return "I couldn't find any relevant context to answer your question."
            
        prompt = self.format_prompt(query, context_chunks)
        
        if self.provider == "gemini":
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                return response.text
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return "Sorry, I encountered an error while generating the response."
        else:
            logger.info("Using dummy generation (No API key found).")
            citations = ", ".join([f"[{i+1}]" for i in range(len(context_chunks))])
            return f"This is a placeholder answer since no LLM API key was provided. The answer would be synthesized using contexts {citations}."
