# generation.py
import ollama


class Generator:
    def __init__(self, model_name="mistral:7b"):
        """
        Args:
            model_name: The Ollama model name (e.g., "mistral", "mistral:7b-instruct")
        """
        self.model_name = model_name

    def build_prompt(self, query, chunks):
        """
        Build a prompt that feeds user query + retrieved chunks into the LLM
        """
        context_text = "\n\n".join(
            f"Source: {c['metadata'].get('filename', 'unknown')} | Page: {c['metadata'].get('page_number', 'N/A')}\n{c['text']}"
            for c in chunks
        )

        prompt = f"""You are a helpful assistant. Use the provided context to answer the question. Context: {context_text}, Question: {query} Answer (be concise, cite sources if relevant):"""
        return prompt

    def generate(self, query, chunks, max_tokens=300, temperature=0.7):
        """
        Generate an answer from query + retrieved chunks using Ollama
        """
        prompt = self.build_prompt(query, chunks)

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": max_tokens},
        )

        return response["response"].strip()
