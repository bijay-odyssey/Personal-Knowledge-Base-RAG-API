from groq import Groq
from typing import List, Dict
import os
import logging
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, provider: str = "groq", model: str = "llama-3.3-70b-versatile"):
        if provider.lower() != "groq":
            raise ValueError("This version only supports Groq provider")

        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        if not self.client.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.model = model
        logger.info(f"Groq Generator initialized with model: {self.model}")

    def generate(self, query: str, contexts: List[Dict[str, str]]) -> str:
        """
        Generate an answer using retrieved contexts via Groq API.
        """
        if not contexts:
            return "No relevant information found in the documents."

        # Build context string for the prompt
        context_str = "\n".join([
            f"- {c['text']} (from {c.get('source', 'unknown')})"
            for c in contexts
        ])

        prompt = (
            "You are a helpful assistant answering questions based only on the provided context. "
            "If the context doesn't contain the answer, say so clearly.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer concisely and accurately:"
        )

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                # Optional: tune these as needed
                temperature=0.3,          # Lower = more deterministic
                max_tokens=1024,
                top_p=0.9,
                # stream=False               
            )

            answer = chat_completion.choices[0].message.content.strip()
            return answer

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return f"Error generating answer: {str(e)}"