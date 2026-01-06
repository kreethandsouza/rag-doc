import os
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

GEN_MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(GEN_MODEL_NAME)

def build_prompt(context_chunks: List[str], question: str) -> str:
    """
    Construct a grounded RAG prompt.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are a biology textbook assistant.

    Answer the question using ONLY the information provided in the context.
    If the answer is not explicitly present, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer (one clear sentence):
    """
    return prompt.strip()


def generate_answer(context_chunks: List[str], question: str) -> str:
    """
    Generate an answer using Gemini grounded on retrieved context.
    """
    prompt = build_prompt(context_chunks, question)

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 256,
        }
    )

    return response.text.strip()


def run_rag(query, index, chunks, retrieve_fn, top_k=5):
    """
    End-to-end RAG execution.
    """
    retrieved_chunks = retrieve_fn(query, index, chunks, top_k=top_k)

    answer = generate_answer(
        context_chunks=retrieved_chunks,
        question=query
    )
    
    return {
        "question": query,
        "context": retrieved_chunks,
        "answer": answer
    }


if __name__ == "__main__":
    from processing_pipeline import (
        load_pdf_context,
        chunk_text,
        build_vector_store,
        retrieve_chunks,
    )

    # Step 1: Prepare data
    text = load_pdf_context()
    chunks = chunk_text(text)
    index = build_vector_store(chunks)

    # Step 2: Ask a question
    query = "What is a molecule?"

    result = run_rag(
        query=query,
        index=index,
        chunks=chunks,
        retrieve_fn=retrieve_chunks,
        top_k=5,
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])
